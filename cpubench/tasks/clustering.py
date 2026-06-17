"""Clustering — scikit-learn (SPEC §5.4)."""

from __future__ import annotations

from cpubench import datasets
from cpubench.registry import task
from cpubench.threading_ctl import resolve_n_jobs


@task(
    "cl_kmeans",
    "clustering",
    data=datasets.blobs,
    sizes={
        "quick": {"n_samples": 100_000, "n_features": 20, "centers": 25, "k": 25},
        "normal": {"n_samples": 600_000, "n_features": 50, "centers": 100, "k": 100},
    },
)
def cl_kmeans(params, ctx):
    from sklearn.cluster import KMeans

    x = ctx.data
    with ctx.timer():
        model = KMeans(n_clusters=int(ctx.params["k"]), n_init=3, max_iter=100, random_state=1337)
        model.fit(x)
    return {"inertia": float(model.inertia_)}


@task(
    "cl_mbkmeans",
    "clustering",
    data=datasets.blobs,
    sizes={
        "quick": {"n_samples": 300_000, "n_features": 20, "centers": 25, "k": 25},
        "normal": {"n_samples": 3_000_000, "n_features": 50, "centers": 100, "k": 100},
    },
)
def cl_mbkmeans(params, ctx):
    from sklearn.cluster import MiniBatchKMeans

    x = ctx.data
    with ctx.timer():
        model = MiniBatchKMeans(
            n_clusters=int(ctx.params["k"]), batch_size=1024, n_init=3, random_state=1337
        )
        model.fit(x)
    return {"inertia": float(model.inertia_)}


@task(
    "cl_optics",
    "clustering",
    data=datasets.blobs,
    sizes={
        "quick": {"n_samples": 4_000, "n_features": 10, "centers": 10},
        "normal": {"n_samples": 30_000, "n_features": 10, "centers": 10},
    },
)
def cl_optics(params, ctx):
    from sklearn.cluster import OPTICS

    x = ctx.data
    with ctx.timer():
        model = OPTICS(
            min_samples=10, max_eps=5.0, algorithm="ball_tree", n_jobs=resolve_n_jobs(ctx.threads)
        )
        model.fit(x)
    n_clusters = int(model.labels_.max()) + 1
    return {"n_clusters": n_clusters, "n_points": int(x.shape[0])}


@task(
    "cl_gmm",
    "clustering",
    data=datasets.blobs,
    sizes={
        "quick": {"n_samples": 50_000, "n_features": 10, "centers": 10, "comp": 10},
        "normal": {"n_samples": 200_000, "n_features": 10, "centers": 30, "comp": 30},
    },
)
def cl_gmm(params, ctx):
    from sklearn.mixture import GaussianMixture

    x = ctx.data
    with ctx.timer():
        model = GaussianMixture(
            n_components=int(ctx.params["comp"]), covariance_type="full", random_state=1337
        )
        model.fit(x)
    return {"converged": bool(model.converged_)}
