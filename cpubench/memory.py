"""Peak-RSS tracking + swap detection around the timed region (SPEC §3).

No pre-task estimate guard and no ``mem_estimate`` — a too-big task simply OOM-fails. Swap
during a task flags ``swapped: true`` (it measures the disk, not the CPU) and is excluded
from scoring downstream.
"""

from __future__ import annotations

import threading
import time

import psutil


class MemorySampler:
    """High-water RSS sampler + swap-in/out delta around the timed region."""

    # Only trust the (system-wide) swap counters when this task was actually memory-bound,
    # i.e. its peak RSS reached this fraction of total RAM. macOS pages/compresses memory
    # almost constantly, so an unconditional sin/sout delta over-flags tiny tasks.
    SWAP_RSS_FRACTION = 0.9

    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self._proc = psutil.Process()
        self.peak_bytes = self._proc.memory_info().rss
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._swap0 = None
        try:
            self._ram_total = psutil.virtual_memory().total
        except Exception:
            self._ram_total = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                rss = self._proc.memory_info().rss
            except psutil.Error:
                break
            if rss > self.peak_bytes:
                self.peak_bytes = rss
            time.sleep(self.interval)

    def start(self) -> None:
        try:
            self._swap0 = psutil.swap_memory()
        except Exception:
            self._swap0 = None
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> tuple[float, bool]:
        """Return ``(peak_rss_mb, swapped)``."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        swapped = False
        memory_bound = (
            self._ram_total is not None
            and self.peak_bytes >= self.SWAP_RSS_FRACTION * self._ram_total
        )
        if self._swap0 is not None and memory_bound:
            try:
                swap1 = psutil.swap_memory()
                swapped = (swap1.sin - self._swap0.sin) > 0 or (swap1.sout - self._swap0.sout) > 0
            except Exception:
                swapped = False
        return self.peak_bytes / (1024 * 1024), swapped
