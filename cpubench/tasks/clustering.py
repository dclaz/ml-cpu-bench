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
        "quick": {"n_samples": 60_000, "n_features": 20, "centers": 25, "k": 25},
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
    "cl_optics",
    "clustering",
    data=datasets.blobs,
    sizes={
        "quick": {"n_samples": 3_000, "n_features": 10, "centers": 10, "max_eps": 2.0},
        "normal": {"n_samples": 30_000, "n_features": 10, "centers": 10, "max_eps": 2.0},
    },
)
def cl_optics(params, ctx):
    from sklearn.cluster import OPTICS

    x = ctx.data
    # max_eps bounds the neighbour search (and reachability): a finite, tight eps with the
    # ball-tree index keeps OPTICS from degenerating toward the full O(n^2) distance work.
    max_eps = float(ctx.params.get("max_eps", 2.0))
    with ctx.timer():
        model = OPTICS(
            min_samples=10,
            max_eps=max_eps,
            algorithm="ball_tree",
            n_jobs=resolve_n_jobs(ctx.threads),
        )
        model.fit(x)
    n_clusters = int(model.labels_.max()) + 1
    return {"n_clusters": n_clusters, "n_points": int(x.shape[0])}


@task(
    "cl_gmm",
    "clustering",
    data=datasets.blobs,
    sizes={
        "quick": {"n_samples": 30_000, "n_features": 10, "centers": 10, "comp": 10},
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


@task(
    "cl_agglom",
    "clustering",
    data=datasets.blobs,
    sizes={
        # Ward agglomerative builds the full hierarchy: ~O(n^2) memory/time with no
        # connectivity constraint, so n is kept small (normal sizes are first guesses).
        "quick": {"n_samples": 2_000, "n_features": 10, "centers": 10, "k": 10},
        "normal": {"n_samples": 20_000, "n_features": 10, "centers": 25, "k": 25},
    },
)
def cl_agglom(params, ctx):
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering

    x = ctx.data
    with ctx.timer():
        model = AgglomerativeClustering(n_clusters=int(ctx.params["k"]), linkage="ward")
        model.fit(x)
    return {"n_clusters": int(model.n_clusters_), "largest": int(np.bincount(model.labels_).max())}
