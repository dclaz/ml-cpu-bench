"""Dense linear algebra — NumPy / SciPy BLAS/LAPACK stress (SPEC §5.2).

All ``backend_sensitive=True``. No destructive flags (``overwrite_*`` off) so the timed region
stays read-only across reps.
"""

from __future__ import annotations

import numpy as np

from cpubench import datasets
from cpubench.registry import task


@task(
    "la_gemm",
    "linalg",
    data=datasets.gemm_pair,
    backend_sensitive=True,
    sizes={"quick": {"N": 2000}, "normal": {"N": 15000}},
)
def la_gemm(params, ctx):
    a, b = ctx.data
    with ctx.timer():
        c = a @ b
    return {"trace": float(np.trace(c))}


@task(
    "la_solve",
    "linalg",
    data=datasets.solve_system,
    backend_sensitive=True,
    sizes={"quick": {"N": 2000}, "normal": {"N": 15000}},
)
def la_solve(params, ctx):
    import scipy.linalg as sla

    a, b = ctx.data
    with ctx.timer():
        x = sla.solve(a, b, overwrite_a=False, overwrite_b=False)
    return {"sum": float(x.sum())}


@task(
    "la_cholesky",
    "linalg",
    data=datasets.spd_matrix,
    backend_sensitive=True,
    sizes={"quick": {"N": 2000}, "normal": {"N": 18000}},
)
def la_cholesky(params, ctx):
    import scipy.linalg as sla

    a = ctx.data
    with ctx.timer():
        ll = sla.cholesky(a, lower=True, overwrite_a=False)
    return {"diag0": float(ll[0, 0])}


@task(
    "la_qr",
    "linalg",
    data=datasets.rect_matrix,
    backend_sensitive=True,
    sizes={"quick": {"rows": 2000, "cols": 1000}, "normal": {"rows": 18000, "cols": 6000}},
)
def la_qr(params, ctx):
    import scipy.linalg as sla

    a = ctx.data
    with ctx.timer():
        q, r = sla.qr(a, mode="economic", overwrite_a=False)
    return {"shape": f"{q.shape[0]}x{r.shape[1]}"}


@task(
    "la_svd",
    "linalg",
    data=datasets.rect_matrix,
    backend_sensitive=True,
    sizes={"quick": {"rows": 2000, "cols": 1000}, "normal": {"rows": 12000, "cols": 4500}},
)
def la_svd(params, ctx):
    a = ctx.data
    with ctx.timer():
        s = np.linalg.svd(a, full_matrices=False, compute_uv=False)
    return {"s0": float(s[0])}


@task(
    "la_eigh",
    "linalg",
    data=datasets.spd_matrix,
    backend_sensitive=True,
    sizes={"quick": {"N": 1500}, "normal": {"N": 8000}},
)
def la_eigh(params, ctx):
    a = ctx.data
    with ctx.timer():
        w = np.linalg.eigvalsh(a)
    return {"max_eig": float(w[-1])}


@task(
    "la_fft",
    "linalg",
    data=datasets.fft_signal,
    backend_sensitive=True,
    sizes={"quick": {"length": 2**22}, "normal": {"length": 2**27}},
)
def la_fft(params, ctx):
    import scipy.fft

    x = ctx.data
    with ctx.timer():
        y = scipy.fft.fft(x)
    return {"abs0": float(abs(y[0]))}
