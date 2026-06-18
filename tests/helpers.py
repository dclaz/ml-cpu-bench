"""Shared test helpers."""

from __future__ import annotations

import numpy as np

from cpubench import registry
from cpubench.runner import RunContext, run_reps

# Keys whose magnitude drives data size/time — clamped tiny for the smoke suite.
_BIG = {"rows", "left", "right", "n_samples", "n_docs", "train_rows"}


def tiny_params(params: dict) -> dict:
    p = dict(params)
    for k in list(p):
        if k in _BIG:
            p[k] = min(p[k], 2000)
    if "length" in p:
        p["length"] = 2**12
    if "n_features" in p:
        p["n_features"] = min(p["n_features"], 500)
    if "n_series" in p:
        p["n_series"] = min(p["n_series"], 6)
    if "series_len" in p:
        p["series_len"] = min(p["series_len"], 64)
    if "entities" in p:
        p["entities"] = min(p["entities"], 100)
    if "vocab" in p:
        p["vocab"] = min(p["vocab"], 500)
    if "groups" in p:
        p["groups"] = min(p["groups"], 100)
    if "cardinality" in p:
        p["cardinality"] = min(p["cardinality"], 1000)
    return p


def run_task_tiny(spec: registry.TaskSpec, engine: str | None, *, repeat: int = 1):
    """Build tiny data and run a task in-process; return (stats, checksum)."""
    params = tiny_params(spec.sizes.get("quick") or next(iter(spec.sizes.values())))
    data = spec.data(params, np.random.default_rng(1337), engine)
    ctx = RunContext(
        data=data, params=params, threads=1, rng=np.random.default_rng(1337), engine=engine
    )
    stats = run_reps(spec.func, ctx, repeat=repeat, warmup=False, offcore_target=[])
    return stats
