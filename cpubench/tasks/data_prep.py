"""Data preparation + feature engineering (SPEC §5.1), each on both pandas and Polars.

The shared frame/panel is built untimed (``ctx.data``); only the operation is inside
``ctx.timer()``. FE is leakage-safe (prior observations only) and builds features into a fresh
structure each call — the shared panel is never mutated.
"""

from __future__ import annotations

import numpy as np

from cpubench import datasets
from cpubench.registry import task

ENGINES = ("pandas", "polars")
_FE_SIZES = {
    "quick": {"rows": 1_200_000, "entities": 2_000},
    "normal": {"rows": 10_000_000, "entities": 20_000},
}


def _shape(obj) -> dict:
    try:
        return {"rows": int(obj.shape[0]), "cols": int(obj.shape[1])}
    except (AttributeError, IndexError):
        return {"rows": int(getattr(obj, "shape", [len(obj)])[0])}


# =================================================================== core data prep
@task(
    "dp_groupby",
    "data_prep",
    data=datasets.core_frame,
    engines=ENGINES,
    sizes={
        "quick": {"rows": 2_000_000, "groups": 1_000},
        "normal": {"rows": 10_000_000, "groups": 50_000},
    },
)
def dp_groupby(params, ctx):
    df = ctx.data
    cols = ["num_0", "num_1", "num_2", "num_3"]
    if ctx.engine == "pandas":
        with ctx.timer():
            out = df.groupby("cat_high")[cols].agg(["sum", "mean", "std", "count"])
        return _shape(out)
    import polars as pl

    aggs = []
    for c in cols:
        aggs += [
            pl.col(c).sum().alias(f"{c}_sum"),
            pl.col(c).mean().alias(f"{c}_mean"),
            pl.col(c).std().alias(f"{c}_std"),
            pl.col(c).count().alias(f"{c}_cnt"),
        ]
    with ctx.timer():
        out = df.group_by("cat_high").agg(aggs)
    return _shape(out)


@task(
    "dp_join",
    "data_prep",
    data=datasets.join_frames,
    engines=ENGINES,
    sizes={
        "quick": {"left": 500_000, "right": 500_000},
        "normal": {"left": 8_000_000, "right": 2_000_000},
    },
)
def dp_join(params, ctx):
    left, right = ctx.data
    if ctx.engine == "pandas":
        with ctx.timer():
            out = left.merge(right, on="key", how="inner")
        return _shape(out)
    with ctx.timer():
        out = left.join(right, on="key", how="inner")
    return _shape(out)


@task(
    "dp_sort",
    "data_prep",
    data=datasets.core_frame,
    engines=ENGINES,
    sizes={
        "quick": {"rows": 500_000, "groups": 25_000},
        "normal": {"rows": 10_000_000, "groups": 200_000},
    },
)
def dp_sort(params, ctx):
    df = ctx.data
    keys = ["cat_high", "num_0"]
    if ctx.engine == "pandas":
        with ctx.timer():  # sort_values returns a new frame (non-mutating)
            out = df.sort_values(keys)
        return _shape(out)
    with ctx.timer():
        out = df.sort(keys)
    return _shape(out)


@task(
    "dp_filter",
    "data_prep",
    data=datasets.core_frame,
    engines=ENGINES,
    sizes={
        "quick": {"rows": 2_000_000, "groups": 1_000},
        "normal": {"rows": 10_000_000, "groups": 50_000},
    },
)
def dp_filter(params, ctx):
    df = ctx.data
    if ctx.engine == "pandas":
        with ctx.timer():
            out = df[df["num_0"] > 0.0][["num_1", "num_2", "cat_low"]]
        return _shape(out)
    import polars as pl

    with ctx.timer():
        out = df.filter(pl.col("num_0") > 0.0).select(["num_1", "num_2", "cat_low"])
    return _shape(out)


@task(
    "dp_string",
    "data_prep",
    data=datasets.string_frame,
    engines=ENGINES,
    sizes={"quick": {"rows": 1_000_000}, "normal": {"rows": 6_000_000}},
)
def dp_string(params, ctx):
    df = ctx.data
    if ctx.engine == "pandas":
        with ctx.timer():
            extracted = df["text"].str.extract(r"ID-(\d+)-", expand=False)
            df["text"].str.lower()  # timed work; result discarded
            mask = df["text"].str.contains("alpha", regex=False)
        return {"rows": int(len(extracted)), "matches": int(mask.sum())}
    import polars as pl

    with ctx.timer():
        out = df.select(
            pl.col("text").str.extract(r"ID-(\d+)-", 1).alias("ex"),
            pl.col("text").str.to_lowercase().alias("lo"),
            pl.col("text").str.contains("alpha").alias("has"),
        )
    return {"rows": int(out.shape[0]), "matches": int(out["has"].sum())}


@task(
    "dp_rolling",
    "data_prep",
    data=datasets.core_frame,
    engines=ENGINES,
    sizes={
        "quick": {"rows": 1_000_000, "groups": 1_000},
        "normal": {"rows": 6_000_000, "groups": 50_000},
    },
)
def dp_rolling(params, ctx):
    df = ctx.data
    if ctx.engine == "pandas":
        with ctx.timer():
            r = df["num_0"].rolling(window=100, min_periods=1)
            out = r.mean(), r.std(), r.median()
        return {"rows": int(len(out[0]))}
    import polars as pl

    with ctx.timer():
        out = df.select(
            pl.col("num_0").rolling_mean(100, min_samples=1).alias("m"),
            pl.col("num_0").rolling_std(100, min_samples=1).alias("s"),
            pl.col("num_0").rolling_median(100, min_samples=1).alias("md"),
        )
    return _shape(out)


# =================================================================== feature engineering
@task("fe_lags", "data_prep", data=datasets.panel_frame, engines=ENGINES, sizes=_FE_SIZES)
def fe_lags(params, ctx):
    df = ctx.data
    if ctx.engine == "pandas":
        import pandas as pd

        g = df.groupby("entity_id")["target"]
        with ctx.timer():
            out = pd.DataFrame({"lag_1": g.shift(1), "lag_7": g.shift(7), "lag_30": g.shift(30)})
        return _shape(out)
    import polars as pl

    with ctx.timer():
        out = df.select(
            pl.col("target").shift(1).over("entity_id").alias("lag_1"),
            pl.col("target").shift(7).over("entity_id").alias("lag_7"),
            pl.col("target").shift(30).over("entity_id").alias("lag_30"),
        )
    return _shape(out)


@task("fe_rolling", "data_prep", data=datasets.panel_frame, engines=ENGINES, sizes=_FE_SIZES)
def fe_rolling(params, ctx):
    df = ctx.data
    if ctx.engine == "pandas":
        import pandas as pd

        g = df.groupby("entity_id")["target"]
        with ctx.timer():
            # closed="left" ⇒ leakage-safe (current row excluded)
            def roll(w, fn):
                r = g.rolling(w, min_periods=1, closed="left")
                return getattr(r, fn)().reset_index(level=0, drop=True)

            out = pd.DataFrame(
                {
                    "m7": roll(7, "mean"),
                    "m30": roll(30, "mean"),
                    "m90": roll(90, "mean"),
                    "s30": roll(30, "std"),
                }
            )
        return _shape(out)
    import polars as pl

    with ctx.timer():
        out = df.select(
            pl.col("target").shift(1).rolling_mean(7, min_samples=1).over("entity_id").alias("m7"),
            pl.col("target")
            .shift(1)
            .rolling_mean(30, min_samples=1)
            .over("entity_id")
            .alias("m30"),
            pl.col("target")
            .shift(1)
            .rolling_mean(90, min_samples=1)
            .over("entity_id")
            .alias("m90"),
            pl.col("target").shift(1).rolling_std(30, min_samples=1).over("entity_id").alias("s30"),
        )
    return _shape(out)


@task("fe_expanding", "data_prep", data=datasets.panel_frame, engines=ENGINES, sizes=_FE_SIZES)
def fe_expanding(params, ctx):
    df = ctx.data
    if ctx.engine == "pandas":
        import pandas as pd

        gid = df["entity_id"]
        with ctx.timer():
            # Vectorized leakage-safe expanding mean of prior obs (matches the Polars
            # cum_sum / cum_count form): prior_mean[k] = mean(target[0..k-1]) per entity.
            shifted = df.groupby("entity_id")["target"].shift(1)
            csum = shifted.groupby(gid).cumsum()
            cnt = shifted.notna().groupby(gid).cumsum()
            prior_mean = csum / cnt.where(cnt > 0)
            norm = df["target"] / prior_mean.replace(0.0, np.nan)
            out = pd.DataFrame({"prior_mean": prior_mean, "norm": norm})
        return _shape(out)
    import polars as pl

    with ctx.timer():
        prior = pl.col("target").shift(1).cum_sum().over("entity_id") / pl.col("target").shift(
            1
        ).cum_count().over("entity_id")
        out = df.select(
            prior.alias("prior_mean"),
            (pl.col("target") / prior).alias("norm"),
        )
    return _shape(out)


@task("fe_ewm", "data_prep", data=datasets.panel_frame, engines=ENGINES, sizes=_FE_SIZES)
def fe_ewm(params, ctx):
    df = ctx.data
    if ctx.engine == "pandas":
        g = df.groupby("entity_id")["target"]
        with ctx.timer():
            out = g.transform(lambda s: s.ewm(span=10, adjust=False).mean())
        return {"rows": int(len(out))}
    import polars as pl

    with ctx.timer():
        out = df.select(
            pl.col("target").ewm_mean(span=10, adjust=False).over("entity_id").alias("ewm")
        )
    return _shape(out)


@task("fe_onehot", "data_prep", data=datasets.panel_frame, engines=ENGINES, sizes=_FE_SIZES)
def fe_onehot(params, ctx):
    df = ctx.data
    if ctx.engine == "pandas":
        import pandas as pd

        sub = df[["cat_low", "cat_med"]]
        with ctx.timer():
            out = pd.get_dummies(sub, dtype=np.uint8)
        assert out.to_numpy().dtype == np.uint8
        return _shape(out)
    import polars as pl

    sub = df.select(["cat_low", "cat_med"])
    with ctx.timer():
        out = sub.to_dummies().select(pl.all().cast(pl.UInt8))
    return _shape(out)


@task("fe_rank", "data_prep", data=datasets.panel_frame, engines=ENGINES, sizes=_FE_SIZES)
def fe_rank(params, ctx):
    df = ctx.data
    if ctx.engine == "pandas":
        import pandas as pd

        g = df.groupby("timestamp")["target"]
        with ctx.timer():
            rank = g.rank()
            mean = g.transform("mean")
            std = g.transform("std").replace(0.0, np.nan)
            z = (df["target"] - mean) / std
            out = pd.DataFrame({"rank": rank, "z": z})
        return _shape(out)
    import polars as pl

    with ctx.timer():
        mean = pl.col("target").mean().over("timestamp")
        std = pl.col("target").std().over("timestamp")
        out = df.select(
            pl.col("target").rank().over("timestamp").alias("rank"),
            ((pl.col("target") - mean) / std).alias("z"),
        )
    return _shape(out)


@task("fe_datetime", "data_prep", data=datasets.panel_frame, engines=ENGINES, sizes=_FE_SIZES)
def fe_datetime(params, ctx):
    df = ctx.data
    if ctx.engine == "pandas":
        import pandas as pd

        with ctx.timer():
            dt = df["timestamp"].dt
            dow = dt.dayofweek.astype(np.int64)
            month = dt.month.astype(np.int64)
            quarter = dt.quarter.astype(np.int64)
            sin = np.sin(2 * np.pi * dow / 7.0)
            cos = np.cos(2 * np.pi * dow / 7.0)
            vec = np.clip(np.log1p(np.abs(df["target"].to_numpy())), 0, 5)
            out = pd.DataFrame(
                {"dow": dow, "month": month, "quarter": quarter, "sin": sin, "cos": cos, "vec": vec}
            )
        return _shape(out)
    import polars as pl

    with ctx.timer():
        dow = pl.col("timestamp").dt.weekday()
        out = df.select(
            dow.alias("dow"),
            pl.col("timestamp").dt.month().alias("month"),
            pl.col("timestamp").dt.quarter().alias("quarter"),
            (2 * np.pi * dow / 7.0).sin().alias("sin"),
            (2 * np.pi * dow / 7.0).cos().alias("cos"),
            pl.col("target").abs().log1p().clip(0, 5).alias("vec"),
        )
    return _shape(out)
