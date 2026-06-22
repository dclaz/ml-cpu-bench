"""Model fitting — scikit-learn + LightGBM (SPEC §5.5).

``copy_X=True`` / ``warm_start=False`` throughout so each fit fully retrains on read-only input.
``n_jobs`` / ``num_threads`` come from ``ctx.threads``. ``md_rf_predict`` trains its forest
untimed (cached across reps) and times only ``predict``.
"""

from __future__ import annotations

import platform

from cpubench import datasets
from cpubench.registry import task
from cpubench.threading_ctl import resolve_n_jobs, resolve_num_threads

_REG_SIZES = {
    "quick": {"n_samples": 25_000, "n_features": 40},
    "normal": {"n_samples": 1_000_000, "n_features": 300},
}
_CLS_SIZES = {
    "quick": {"n_samples": 200_000, "n_features": 50, "n_classes": 3},
    "normal": {"n_samples": 800_000, "n_features": 150, "n_classes": 3},
}


def _import_lightgbm():
    try:
        import lightgbm as lgb

        return lgb
    except Exception as exc:  # pragma: no cover - platform/runtime dependent
        msg = "LightGBM import failed."
        if platform.system() == "Darwin":
            msg += (
                " On macOS, this is usually a missing OpenMP runtime. "
                "Install it with: brew install libomp"
            )
        raise RuntimeError(msg) from exc


@task("md_linreg", "models", data=datasets.regression_xy, sizes=_REG_SIZES)
def md_linreg(params, ctx):
    from sklearn.linear_model import LinearRegression

    x, y = ctx.data
    with ctx.timer():
        model = LinearRegression(copy_X=True).fit(x, y)
    return {"coef0": float(model.coef_[0])}


@task("md_ridge", "models", data=datasets.regression_xy, sizes=_REG_SIZES)
def md_ridge(params, ctx):
    from sklearn.linear_model import RidgeCV

    x, y = ctx.data
    with ctx.timer():
        model = RidgeCV(alphas=(0.1, 1.0, 10.0)).fit(x, y)
    return {"alpha": float(model.alpha_)}


@task("md_lasso", "models", data=datasets.regression_xy, sizes=_REG_SIZES)
def md_lasso(params, ctx):
    from sklearn.linear_model import LassoCV

    x, y = ctx.data
    with ctx.timer():
        model = LassoCV(
            alphas=10, cv=3, copy_X=True, n_jobs=resolve_n_jobs(ctx.threads), random_state=1337
        ).fit(x, y)
    return {"alpha": float(model.alpha_)}


@task("md_logreg", "models", data=datasets.classification_xy, sizes=_CLS_SIZES)
def md_logreg(params, ctx):
    from sklearn.linear_model import LogisticRegression

    x, y = ctx.data
    with ctx.timer():
        model = LogisticRegression(solver="lbfgs", max_iter=100).fit(x, y)
    return {"n_iter": int(model.n_iter_[0])}


@task("md_bayes_ridge", "models", data=datasets.regression_xy, sizes=_REG_SIZES)
def md_bayes_ridge(params, ctx):
    from sklearn.linear_model import BayesianRidge

    x, y = ctx.data
    with ctx.timer():
        model = BayesianRidge().fit(x, y)
    return {"alpha": float(model.alpha_)}


@task(
    "md_gpr",
    "models",
    data=datasets.regression_xy,
    backend_sensitive=True,
    modes=("normal",),
    sizes={"normal": {"n_samples": 10_000, "n_features": 20}},
)
def md_gpr(params, ctx):
    import numpy as np
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF

    x, y = ctx.data
    with ctx.timer():
        model = GaussianProcessRegressor(kernel=RBF(), optimizer=None, copy_X_train=True)
        model.fit(x, y)
        lml = model.log_marginal_likelihood()
    return {"lml": float(np.round(lml, 2))}


@task(
    "md_rf",
    "models",
    data=datasets.classification_xy,
    sizes={
        "quick": {"n_samples": 25_000, "n_features": 50, "n_classes": 3, "trees": 40},
        "normal": {"n_samples": 200_000, "n_features": 50, "n_classes": 3, "trees": 300},
    },
)
def md_rf(params, ctx):
    from sklearn.ensemble import RandomForestClassifier

    x, y = ctx.data
    with ctx.timer():
        model = RandomForestClassifier(
            n_estimators=int(ctx.params["trees"]),
            n_jobs=resolve_n_jobs(ctx.threads),
            random_state=1337,
        )
        model.fit(x, y)
    return {"trees": int(ctx.params["trees"])}


@task(
    "md_rf_predict",
    "models",
    data=datasets.rf_predict_data,
    sizes={
        "quick": {"rows": 500_000, "n_features": 50, "train_rows": 50_000, "trees": 80},
        "normal": {"rows": 5_000_000, "n_features": 50, "train_rows": 200_000, "trees": 300},
    },
)
def md_rf_predict(params, ctx):
    from sklearn.ensemble import RandomForestClassifier

    x_tr, y_tr, x_te = ctx.data
    if "forest" not in ctx.cache:  # train untimed, once, reuse across reps
        forest = RandomForestClassifier(
            n_estimators=int(ctx.params["trees"]),
            n_jobs=resolve_n_jobs(ctx.threads),
            random_state=1337,
        )
        forest.fit(x_tr, y_tr)
        ctx.cache["forest"] = forest
    forest = ctx.cache["forest"]
    with ctx.timer():
        preds = forest.predict(x_te)
    return {"rows": int(x_te.shape[0]), "prediction_sum": int(preds.sum())}


@task(
    "md_lgbm",
    "models",
    data=datasets.regression_xy,
    sizes={
        "quick": {"n_samples": 20_000, "n_features": 50, "trees": 40},
        "normal": {"n_samples": 500_000, "n_features": 100, "trees": 500},
    },
)
def md_lgbm(params, ctx):
    lgb = _import_lightgbm()

    x, y = ctx.data
    with ctx.timer():
        lgb.LGBMRegressor(
            n_estimators=int(ctx.params["trees"]),
            num_threads=resolve_num_threads(ctx.threads),
            random_state=1337,
            verbose=-1,
        ).fit(x, y)
    return {"trees": int(ctx.params["trees"])}


@task(
    "md_svc_rbf",
    "models",
    data=datasets.classification_xy,
    sizes={
        "quick": {"n_samples": 5_000, "n_features": 30, "n_classes": 3},
        "normal": {"n_samples": 15_000, "n_features": 30, "n_classes": 3},
    },
)
def md_svc_rbf(params, ctx):
    from sklearn.svm import SVC

    x, y = ctx.data
    with ctx.timer():
        model = SVC(kernel="rbf", random_state=1337).fit(x, y)
    return {"n_sv": int(model.support_.shape[0])}


@task(
    "md_autoarima",
    "models",
    data=datasets.arima_panel,
    sizes={
        "quick": {"n_series": 4, "series_len": 96, "horizon": 1},
        "normal": {"n_series": 256, "series_len": 512, "horizon": 12},
    },
)
def md_autoarima(params, ctx):
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA

    train_df, future_df = ctx.data
    horizon = int(ctx.params["horizon"])
    with ctx.timer():
        sf = StatsForecast(
            models=[AutoARIMA(season_length=1)],
            freq=1,
            n_jobs=resolve_n_jobs(ctx.threads),
        )
        fc = sf.forecast(df=train_df, h=horizon, X_df=future_df)
    return {
        "n_series": int(ctx.params["n_series"]),
        "forecast_sum": round(float(fc["AutoARIMA"].fillna(0.0).sum()), 3),
    }


@task(
    "md_knn",
    "models",
    data=datasets.classification_xy,
    sizes={
        "quick": {"n_samples": 50_000, "n_features": 50, "n_classes": 3},
        "normal": {"n_samples": 200_000, "n_features": 50, "n_classes": 3},
    },
)
def md_knn(params, ctx):
    from sklearn.neighbors import KNeighborsClassifier

    x, y = ctx.data
    x_query = x[:1000]
    with ctx.timer():
        model = KNeighborsClassifier(
            n_neighbors=5, algorithm="brute", n_jobs=resolve_n_jobs(ctx.threads)
        ).fit(x, y)
        preds = model.predict(x_query)
    return {"queries": int(preds.shape[0])}
