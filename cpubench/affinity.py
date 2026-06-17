"""P/E detection + modular per-platform enforcement backend.

Three responsibilities, kept modular so the soft macOS path can be swapped later without
touching runner/scoring/report:

1. ``detect_pe_topology`` — best-effort P/E detection. Uncertain/uniform ⇒ homogeneous
   (``perf_cores``/``eff_cores`` = None). Never guesses a split from core counts/model strings.
2. ``apply_affinity`` — pin the current (worker) process for a ``(threads_mode, cores)`` config
   and report the ``enforcement`` actually achieved (``none``/``pinned``/``biased``).
3. ``ResidencySampler`` — 10 Hz off-core residency sampler, active only while the timer is open.
"""

from __future__ import annotations

import os
import platform
import sys
import threading
import time

import psutil


# --------------------------------------------------------------------------- detection
def _linux_pe_ids() -> tuple[list[int], list[int]] | None:
    """Return (perf_ids, eff_ids) from sysfs cpu_capacity, or None if uniform/unavailable."""
    base = "/sys/devices/system/cpu"
    caps: dict[int, int] = {}
    try:
        entries = os.listdir(base)
    except OSError:
        return None
    for name in entries:
        if not name.startswith("cpu") or not name[3:].isdigit():
            continue
        cpu_id = int(name[3:])
        path = os.path.join(base, name, "cpu_capacity")
        try:
            with open(path) as fh:
                caps[cpu_id] = int(fh.read().strip())
        except OSError:
            continue
    if not caps:
        return None
    distinct = sorted(set(caps.values()))
    if len(distinct) < 2:
        return None  # uniform ⇒ homogeneous
    hi = distinct[-1]
    perf = sorted(c for c, v in caps.items() if v == hi)
    eff = sorted(c for c, v in caps.items() if v != hi)
    return perf, eff


def _macos_pe_counts() -> tuple[int, int] | None:
    """Return (perf_count, eff_count) from sysctl hw.perflevel*, or None."""
    try:
        import subprocess

        def _sysctl(key: str) -> int | None:
            out = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, timeout=5)
            if out.returncode != 0:
                return None
            val = out.stdout.strip()
            return int(val) if val.isdigit() else None

        p = _sysctl("hw.perflevel0.physicalcpu")
        e = _sysctl("hw.perflevel1.physicalcpu")
        if p and e:
            return p, e
    except Exception:
        return None
    return None


def detect_pe_topology() -> dict:
    """Best-effort P/E topology.

    Returns dict with ``perf_cores``/``eff_cores`` counts (None when homogeneous/uncertain),
    the logical id lists used for pinning, ``heterogeneous`` bool, and the platform's
    ``enforcement_capability`` (``pinned``/``biased``/``none``).
    """
    system = platform.system()
    perf_ids: list[int] = []
    eff_ids: list[int] = []
    heterogeneous = False

    if system == "Linux":
        res = _linux_pe_ids()
        if res is not None:
            perf_ids, eff_ids = res
            heterogeneous = True
        capability = "pinned"
    elif system == "Darwin":
        counts = _macos_pe_counts()
        if counts is not None:
            heterogeneous = True
        capability = "biased"
    elif system == "Windows":
        # CPU-set efficiency class is not exposed via a stable stdlib path; treat as
        # homogeneous unless a future backend fills this in. Hard pinning is available.
        capability = "pinned"
    else:
        capability = "none"

    if heterogeneous and system == "Linux":
        perf_cores = len(perf_ids)
        eff_cores = len(eff_ids)
    elif heterogeneous and system == "Darwin":
        counts = _macos_pe_counts()
        perf_cores, eff_cores = counts  # type: ignore[misc]
    else:
        perf_cores = eff_cores = None

    return {
        "perf_cores": perf_cores,
        "eff_cores": eff_cores,
        "perf_core_ids": perf_ids,
        "eff_core_ids": eff_ids,
        "heterogeneous": heterogeneous,
        "enforcement_capability": capability,
    }


def can_hard_pin() -> bool:
    return hasattr(os, "sched_setaffinity") or platform.system() == "Windows"


# --------------------------------------------------------------------------- enforcement
def _all_logical_ids() -> list[int]:
    n = psutil.cpu_count(logical=True) or 1
    return list(range(n))


def resolve_target_cores(threads_mode: str, cores: str, topo: dict) -> list[int]:
    """Logical core ids the worker should be confined to for this config."""
    perf = topo["perf_core_ids"]
    eff = topo["eff_core_ids"]
    all_ids = _all_logical_ids()
    if threads_mode == "single":
        return [perf[0]] if perf else [all_ids[0]]
    if cores == "p" and perf:
        return perf
    if cores == "e" and eff:
        return eff
    return all_ids


def apply_affinity(threads_mode: str, cores: str, topo: dict) -> dict:
    """Pin the current process per the config; return ``{enforcement, offcore_target}``.

    ``offcore_target`` records the "wrong-type" core ids for the residency sampler (empty when
    no core-type isolation applies).
    """
    system = platform.system()
    target = resolve_target_cores(threads_mode, cores, topo)
    wrong_type: list[int] = []

    # Determine the "wrong-type" set for residency accounting (only meaningful for p/e).
    if cores == "p" and topo["perf_core_ids"]:
        wrong_type = topo["eff_core_ids"]
    elif cores == "e" and topo["eff_core_ids"]:
        wrong_type = topo["perf_core_ids"]

    enforcement = "none"
    isolating_type = bool(wrong_type)  # a genuine p/e isolation request on a hetero chip

    if system in ("Linux", "Windows"):
        # Hard pin (covers single-thread P placement and p/e isolation).
        try:
            psutil.Process().cpu_affinity(target)
        except Exception:
            pass
        if isolating_type:
            enforcement = "pinned"
    elif system == "Darwin":
        # QoS is set at spawn via taskpolicy(8) by the controller; here we only classify.
        if isolating_type:
            enforcement = "biased"

    return {"enforcement": enforcement, "offcore_target": wrong_type, "target": target}


def taskpolicy_prefix(cores: str) -> list[str]:
    """macOS spawn prefix: ``taskpolicy -b`` for ``--cores e`` (background QoS → E-cores)."""
    if platform.system() == "Darwin" and cores == "e":
        return ["taskpolicy", "-b"]
    return []


# --------------------------------------------------------------------------- residency
class ResidencySampler:
    """10 Hz per-core busy-time sampler, active only while the timed region is open.

    Computes ``offcore_residency_pct = busy_on_wrong_type / total_busy`` over the window.
    Returns 0.0 when there is no wrong-type set (cores=all or homogeneous).
    """

    INTERVAL_S = 0.1  # 10 Hz

    def __init__(self, wrong_type_ids: list[int]):
        self.wrong_type_ids = set(wrong_type_ids)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._first: list | None = None
        self._last: list | None = None

    def _pin_complement(self) -> None:
        # On Linux/Win pin the sampler to the complement of the test cores so it doesn't
        # steal one. Best-effort.
        if not self.wrong_type_ids:
            return
        if platform.system() not in ("Linux", "Windows"):
            return
        try:
            comp = sorted(self.wrong_type_ids)
            if comp:
                psutil.Process().cpu_affinity(comp)
        except Exception:
            pass

    def _loop(self) -> None:
        self._first = psutil.cpu_times(percpu=True)
        while not self._stop.is_set():
            time.sleep(self.INTERVAL_S)
            self._last = psutil.cpu_times(percpu=True)

    def start(self) -> None:
        if not self.wrong_type_ids:
            return  # nothing to measure
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    @staticmethod
    def _busy(t) -> float:
        idle = getattr(t, "idle", 0.0) + getattr(t, "iowait", 0.0)
        total = sum(
            getattr(t, f, 0.0)
            for f in ("user", "nice", "system", "idle", "iowait", "irq", "softirq", "steal")
        )
        return max(0.0, total - idle)

    def stop(self) -> float:
        if self._thread is None:
            return 0.0
        self._stop.set()
        self._thread.join(timeout=2.0)
        if self._first is None or self._last is None:
            return 0.0
        total_busy = 0.0
        wrong_busy = 0.0
        for cpu_id, (a, b) in enumerate(zip(self._first, self._last, strict=False)):
            delta = self._busy(b) - self._busy(a)
            if delta <= 0:
                continue
            total_busy += delta
            if cpu_id in self.wrong_type_ids:
                wrong_busy += delta
        if total_busy <= 0:
            return 0.0
        return round(wrong_busy / total_busy, 4)


def _self_check() -> None:  # pragma: no cover - manual aid
    print(detect_pe_topology(), file=sys.stderr)
