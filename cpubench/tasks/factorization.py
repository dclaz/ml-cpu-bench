"""Matrix factorization — scikit-learn (SPEC §5.3). backend_sensitive=True."""

from __future__ import annotations

from cpubench import datasets
from cpubench.registry import task

_DENSE_SIZES = {
    "quick": {"n_samples": 8_000, "n_features": 500},
    "normal": {"n_samples": 100_000, "n_features": 2_000},
}


@task(
    "mf_pca",
    "factorization",
    data=datasets.dense_matrix,
    backend_sensitive=True,
    sizes=_DENSE_SIZES,
)
def mf_pca(params, ctx):
    from sklearn.decomposition import PCA

    x = ctx.data
    with ctx.timer():
        model = PCA(n_components=50, random_state=1337)
        out = model.fit_transform(x)
    return {"shape": f"{out.shape[0]}x{out.shape[1]}"}


@task(
    "mf_nmf",
    "factorization",
    data=datasets.dense_nonneg,
    backend_sensitive=True,
    sizes={
        "quick": {"n_samples": 1000, "n_features": 50},
        "normal": {"n_samples": 20_000, "n_features": 2_000},
    },
)
def mf_nmf(params, ctx):
    from sklearn.decomposition import NMF

    x = ctx.data
    with ctx.timer():
        model = NMF(n_components=20, init="nndsvda", max_iter=200, random_state=1337)
        out = model.fit_transform(x)
    return {"shape": f"{out.shape[0]}x{out.shape[1]}"}
