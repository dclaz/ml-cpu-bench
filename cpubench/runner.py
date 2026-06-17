"""Warm-up + fixed-repetition timing with a read-only timed region (SPEC §3).

Invariants: only the algorithm under test is inside ``ctx.timer()``; warm-up is one discarded
run (default on); exactly ``repeat`` timed reps, no adaptive logic; no reset between reps; the
timed region treats its input as read-only.
"""

from __future__ import annotations

import statistics
import time
from contextlib import contextmanager

from cpubench.affinity import ResidencySampler
from cpubench.memory import MemorySampler


class RunContext:
    """The ``ctx`` handed to a task: ``data``, ``timer()``, ``params``, ``threads``, ``rng``."""

    def __init__(self, *, data, params, threads, rng, engine=None):
        self.data = data
        self.params = params
        self.threads = threads
        self.rng = rng
        self.engine = engine
        self.cache: dict = {}  # untimed setup reused across reps (e.g. a pre-trained model)
        self._wall: list[float] = []
        self._cpu: list[float] = []

    @contextmanager
    def timer(self):
        c0 = time.process_time()
        w0 = time.perf_counter()
        try:
            yield
        finally:
            w1 = time.perf_counter()
            c1 = time.process_time()
            self._wall.append(w1 - w0)
            self._cpu.append(c1 - c0)

    def _reset_timings(self) -> None:
        self._wall.clear()
        self._cpu.clear()


def _cv(median: float, std: float) -> float:
    if median <= 0:
        return 0.0
    return std / median


def run_reps(func, ctx: RunContext, *, repeat: int, warmup: bool, offcore_target: list[int]):
    """Run warm-up (optional) + ``repeat`` timed reps; return a stats dict.

    ``func(params, ctx)`` is expected to open ``ctx.timer()`` exactly once per call and may
    return an informational checksum.
    """
    checksum = None

    if warmup:
        checksum = func(ctx.params, ctx)
        ctx._reset_timings()  # discard the warm-up timing

    mem = MemorySampler()
    res = ResidencySampler(offcore_target)
    mem.start()
    res.start()
    for _ in range(repeat):
        checksum = func(ctx.params, ctx)
    offcore_pct = res.stop()
    peak_rss_mb, swapped = mem.stop()

    reps_s = list(ctx._wall)
    cpu_s = list(ctx._cpu)
    if len(reps_s) != repeat:
        raise RuntimeError(f"task opened ctx.timer() {len(reps_s)} times, expected {repeat}")

    median_s = statistics.median(reps_s)
    min_s = min(reps_s)
    std_s = statistics.pstdev(reps_s) if len(reps_s) > 1 else 0.0
    cv = _cv(median_s, std_s)
    total_wall = sum(reps_s)
    total_cpu = sum(cpu_s)
    cpu_wall_ratio = (total_cpu / total_wall) if total_wall > 0 else 0.0

    return {
        "reps_s": [round(x, 6) for x in reps_s],
        "median_s": median_s,
        "min_s": min_s,
        "std_s": std_s,
        "cv": cv,
        "noisy": cv > 0.10,
        "peak_rss_mb": round(peak_rss_mb, 1),
        "cpu_wall_ratio": round(cpu_wall_ratio, 2),
        "swapped": swapped,
        "offcore_residency_pct": offcore_pct,
        "checksum": checksum,
    }
