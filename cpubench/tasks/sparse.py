"""Sparse & NLP workloads — SciPy sparse + scikit-learn (SPEC §5.6).

Memory-bound, irregular access, largely independent of the dense BLAS backend (backend-neutral
for compare). The hashing-trick transforms are stateless and single-threaded in sklearn.
"""

from __future__ import annotations

from cpubench import datasets
from cpubench.registry import task
from cpubench.threading_ctl import resolve_n_jobs


@task(
    "sp_tfidf",
    "sparse",
    data=datasets.token_corpus,
    sizes={
        "quick": {"n_docs": 50_000, "vocab": 30_000, "tokens_per_doc": 80},
        "normal": {"n_docs": 300_000, "vocab": 30_000, "tokens_per_doc": 80},
    },
)
def sp_tfidf(params, ctx):
    from sklearn.feature_extraction.text import TfidfVectorizer

    docs = ctx.data
    with ctx.timer():
        out = TfidfVectorizer().fit_transform(docs)
    return {"shape": f"{out.shape[0]}x{out.shape[1]}", "nnz": int(out.nnz)}


@task(
    "sp_hashvec",
    "sparse",
    data=datasets.token_corpus,
    sizes={
        "quick": {"n_docs": 50_000, "vocab": 30_000, "tokens_per_doc": 80},
        "normal": {"n_docs": 300_000, "vocab": 30_000, "tokens_per_doc": 80},
    },
)
def sp_hashvec(params, ctx):
    from sklearn.feature_extraction.text import HashingVectorizer

    docs = ctx.data
    vec = HashingVectorizer(n_features=2**20)
    with ctx.timer():
        out = vec.transform(docs)
    return {"shape": f"{out.shape[0]}x{out.shape[1]}", "nnz": int(out.nnz)}


@task(
    "sp_fhash",
    "sparse",
    data=datasets.fhash_rows,
    sizes={
        "quick": {"n_samples": 200_000, "n_fields": 20, "cardinality": 1_000_000},
        "normal": {"n_samples": 1_000_000, "n_fields": 30, "cardinality": 1_000_000},
    },
)
def sp_fhash(params, ctx):
    from sklearn.feature_extraction import FeatureHasher

    rows = ctx.data
    hasher = FeatureHasher(n_features=2**20, input_type="string")
    with ctx.timer():
        out = hasher.transform(rows)
    return {"shape": f"{out.shape[0]}x{out.shape[1]}", "nnz": int(out.nnz)}


@task(
    "sp_matmul",
    "sparse",
    data=datasets.sparse_matmul_data,
    sizes={
        "quick": {"n_samples": 100_000, "n_features": 30_000, "density": 0.002, "rhs_cols": 64},
        "normal": {"n_samples": 200_000, "n_features": 50_000, "density": 0.002, "rhs_cols": 64},
    },
)
def sp_matmul(params, ctx):
    a, b = ctx.data
    with ctx.timer():
        out = a @ b
    return {"shape": f"{out.shape[0]}x{out.shape[1]}"}


@task(
    "sp_tsvd",
    "sparse",
    data=datasets.sparse_csr,
    sizes={
        "quick": {"n_samples": 50_000, "n_features": 10_000, "density": 0.003},
        "normal": {"n_samples": 200_000, "n_features": 50_000, "density": 0.003},
    },
)
def sp_tsvd(params, ctx):
    from sklearn.decomposition import TruncatedSVD

    x = ctx.data
    with ctx.timer():
        out = TruncatedSVD(
            n_components=100, algorithm="randomized", random_state=1337
        ).fit_transform(x)
    return {"shape": f"{out.shape[0]}x{out.shape[1]}"}


@task(
    "sp_nmf",
    "sparse",
    data=datasets.doc_term_tfidf,
    sizes={
        "quick": {"n_samples": 20_000, "n_features": 10_000, "density": 0.004},
        "normal": {"n_samples": 100_000, "n_features": 30_000, "density": 0.004},
    },
)
def sp_nmf(params, ctx):
    from sklearn.decomposition import NMF

    x = ctx.data
    with ctx.timer():
        out = NMF(n_components=20, init="nndsvda", max_iter=200, random_state=1337).fit_transform(x)
    return {"shape": f"{out.shape[0]}x{out.shape[1]}"}


@task(
    "sp_saga",
    "sparse",
    data=datasets.sparse_binary,
    sizes={
        "quick": {"n_samples": 50_000, "n_features": 10_000, "density": 0.003},
        "normal": {"n_samples": 200_000, "n_features": 50_000, "density": 0.003},
    },
)
def sp_saga(params, ctx):
    from sklearn.linear_model import LogisticRegression

    x, y = ctx.data
    with ctx.timer():
        model = LogisticRegression(solver="saga", max_iter=20).fit(x, y)
    return {"n_iter": int(model.n_iter_[0])}


@task(
    "nlp_lda",
    "sparse",
    data=datasets.doc_term_counts,
    modes=("normal",),
    sizes={"normal": {"n_samples": 100_000, "n_features": 30_000, "density": 0.004}},
)
def nlp_lda(params, ctx):
    from sklearn.decomposition import LatentDirichletAllocation

    x = ctx.data
    with ctx.timer():
        model = LatentDirichletAllocation(
            n_components=20,
            learning_method="batch",
            max_iter=10,
            n_jobs=resolve_n_jobs(ctx.threads),
            random_state=1337,
        ).fit(x)
    return {"perplexity": round(float(model.bound_), 2)}


@task(
    "sp_lasso_cv",
    "sparse",
    data=datasets.sparse_regression,
    modes=("normal",),
    sizes={"normal": {"n_samples": 100_000, "n_features": 10_000, "density": 0.005}},
)
def sp_lasso_cv(params, ctx):
    from sklearn.linear_model import LassoCV

    x, y = ctx.data
    with ctx.timer():
        model = LassoCV(cv=5, alphas=50, n_jobs=resolve_n_jobs(ctx.threads), random_state=1337).fit(
            x, y
        )
    return {"alpha": float(model.alpha_)}
