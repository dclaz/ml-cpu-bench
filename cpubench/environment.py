"""Pure environment detection (no side effects).

Detects CPU model/arch, core counts, RAM, OS, Python, library versions, and the BLAS
backend. P/E detection itself lives in ``affinity.py`` (Layer 2); this module consumes
``perf_cores``/``eff_cores`` from it. Returns a dict matching the SPEC §7.1 ``environment``
block.
"""

from __future__ import annotations

import platform
import sys
from importlib import metadata

import psutil

_LIBS = ("numpy", "scipy", "scikit-learn", "pandas", "polars", "lightgbm")


def _lib_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _cpu_model() -> str:
    # platform.processor() is often empty/generic; fall back to OS-specific sources.
    model = platform.processor()
    if model and model != platform.machine():
        return model
    try:  # Linux
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    if platform.system() == "Darwin":  # macOS: marketing name, not the generic arch string
        try:
            import subprocess

            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
        except Exception:
            pass
    return platform.machine() or "unknown"


def detect_blas() -> tuple[str | None, int | None]:
    """Return ``(blas_backend, blas_threads_detected)``.

    Prefers ``threadpoolctl`` (which loads numpy/scipy and inspects the live shared
    libraries); falls back to ``numpy.show_config()`` string scraping. Accelerate on
    macOS-arm64 is the finickiest — kept in one place with clear fallbacks.
    """
    backend: str | None = None
    threads: int | None = None
    try:
        import numpy  # noqa: F401  -- ensure the BLAS shared lib is loaded before scanning
        import threadpoolctl

        # Touch a tiny BLAS op so the backend lib is resident for threadpool_info().
        numpy.dot(numpy.ones(2), numpy.ones(2))
        for pool in threadpoolctl.threadpool_info():
            if pool.get("user_api") == "blas":
                internal = (pool.get("internal_api") or "").lower()
                if "mkl" in internal:
                    backend = "MKL"
                elif "openblas" in internal:
                    backend = "OpenBLAS"
                elif "accelerate" in internal or "veclib" in internal:
                    backend = "Accelerate"
                else:
                    backend = pool.get("internal_api")
                threads = pool.get("num_threads")
                break
    except Exception:
        pass

    if backend is None:
        try:
            import contextlib
            import io

            import numpy as np

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                np.show_config()
            text = buf.getvalue().lower()
            if "mkl" in text:
                backend = "MKL"
            elif "accelerate" in text:
                backend = "Accelerate"
            elif "openblas" in text:
                backend = "OpenBLAS"
        except Exception:
            pass

    return backend, threads


def baseline_load_pct() -> float:
    """System-wide CPU load sampled over a short interval (pre-flight quiet-machine check)."""
    return float(psutil.cpu_percent(interval=0.3))


def detect_environment(
    *,
    perf_cores: int | None = None,
    eff_cores: int | None = None,
    sample_load: bool = True,
) -> dict:
    """Build the SPEC §7.1 ``environment`` dict.

    ``perf_cores``/``eff_cores`` come from ``affinity.py`` (null when homogeneous/uncertain).
    ``import_time_s`` and ``continuous_run`` are filled later by the controller/worker.
    """
    blas_backend, blas_threads = detect_blas()
    vm = psutil.virtual_memory()
    libs = {name: _lib_version(name) for name in _LIBS}

    return {
        "cpu_model": _cpu_model(),
        "arch": platform.machine(),
        "logical_cores": psutil.cpu_count(logical=True),
        "physical_cores": psutil.cpu_count(logical=False),
        "perf_cores": perf_cores,
        "eff_cores": eff_cores,
        "ram_gb": round(vm.total / (1024**3), 1),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "blas_backend": blas_backend,
        "blas_threads_detected": blas_threads,
        "baseline_load_pct": baseline_load_pct() if sample_load else None,
        "import_time_s": None,
        "libs": libs,
        "continuous_run": True,
    }


def python_full() -> str:
    return sys.version.split()[0]
