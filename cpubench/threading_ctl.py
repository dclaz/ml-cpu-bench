"""Thread-count control.

``set_thread_env`` MUST be called by the worker *before any heavy import* so every parallel
library honours one thread count. ``threadpool_limits`` is a belt-and-braces guard around BLAS
regions. ``resolve_n_jobs`` / ``resolve_num_threads`` translate the count for sklearn / lgbm.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

# Env vars that pin thread count across the whole stack. POLARS_MAX_THREADS is critical:
# Polars defaults to *logical* cores, which would unfairly give it more threads than the rest.
_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_MAX_THREADS",
    "POLARS_MAX_THREADS",
)


def set_thread_env(n: int) -> None:
    """Pin every parallel library to ``n`` threads via environment variables.

    Must run before numpy/scipy/sklearn/polars/lightgbm are imported in the worker.
    """
    value = str(int(n))
    for var in _THREAD_ENV_VARS:
        os.environ[var] = value


@contextmanager
def threadpool_limits(n: int):
    """Best-effort context limiting BLAS/OpenMP pools to ``n`` threads inside a region."""
    try:
        import threadpoolctl

        with threadpoolctl.threadpool_limits(limits=int(n)):
            yield
    except Exception:
        yield


def resolve_n_jobs(threads: int) -> int:
    """sklearn ``n_jobs`` from the chosen thread count (1 → 1, else the count)."""
    return int(threads)


def resolve_num_threads(threads: int) -> int:
    """LightGBM ``num_threads`` from the chosen thread count."""
    return int(threads)
