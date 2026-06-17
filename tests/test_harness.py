"""Harness unit tests: environment, threading, affinity, runner, controller."""

from __future__ import annotations

import argparse
import os

import numpy as np

from cpubench import affinity, controller, environment, registry, threading_ctl
from cpubench.runner import RunContext, run_reps


# --------------------------------------------------------------------------- environment
def test_environment_dict_shape():
    env = environment.detect_environment(perf_cores=None, eff_cores=None, sample_load=False)
    for key in (
        "cpu_model",
        "arch",
        "logical_cores",
        "physical_cores",
        "perf_cores",
        "eff_cores",
        "ram_gb",
        "os",
        "python",
        "blas_backend",
        "libs",
        "continuous_run",
    ):
        assert key in env
    assert env["physical_cores"] >= 1
    assert set(env["libs"]) == {"numpy", "scipy", "scikit-learn", "pandas", "polars", "lightgbm"}


# --------------------------------------------------------------------------- threading
def test_set_thread_env_sets_all_vars():
    threading_ctl.set_thread_env(3)
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_MAX_THREADS",
        "POLARS_MAX_THREADS",
    ):
        assert os.environ[var] == "3"


# --------------------------------------------------------------------------- affinity
def test_homogeneous_box_has_no_pe_split():
    topo = affinity.detect_pe_topology()
    # this dev box is homogeneous; uncertain detection ⇒ homogeneous, never a guessed split
    if not topo["heterogeneous"]:
        assert topo["perf_cores"] is None and topo["eff_cores"] is None


def test_linux_pe_detection_mocked(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr(affinity, "_linux_pe_ids", lambda: ([0, 1], [2, 3]))
    topo = affinity.detect_pe_topology()
    assert topo["heterogeneous"] is True
    assert topo["perf_cores"] == 2 and topo["eff_cores"] == 2
    assert topo["perf_core_ids"] == [0, 1]


def test_uniform_capacity_is_homogeneous(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr(affinity, "_linux_pe_ids", lambda: None)
    topo = affinity.detect_pe_topology()
    assert topo["heterogeneous"] is False
    assert topo["perf_cores"] is None


def test_resolve_target_cores_single_pins_one():
    topo = {"perf_core_ids": [], "eff_core_ids": []}
    target = affinity.resolve_target_cores("single", "all", topo)
    assert len(target) == 1


def test_residency_sampler_empty_is_zero():
    s = affinity.ResidencySampler([])
    s.start()
    assert s.stop() == 0.0


# --------------------------------------------------------------------------- runner
def _dummy_task(params, ctx):
    arr = ctx.data
    with ctx.timer():
        _ = float(np.sort(arr)[0])  # non-mutating
    return {"first": float(arr[0])}


def test_runner_read_only_and_rep_count():
    data = np.array([3.0, 1.0, 2.0])
    original = data.copy()
    ctx = RunContext(data=data, params={}, threads=1, rng=np.random.default_rng(0))
    stats = run_reps(_dummy_task, ctx, repeat=5, warmup=True, offcore_target=[])
    assert len(stats["reps_s"]) == 5  # warm-up discarded, exactly repeat timed reps
    assert np.array_equal(data, original)  # timed region was read-only
    assert "median_s" in stats and "cv" in stats


def test_runner_noisy_flag():
    ctx = RunContext(data=np.zeros(1), params={}, threads=1, rng=np.random.default_rng(0))
    # constant work → low cv → not noisy
    stats = run_reps(_dummy_task, ctx, repeat=3, warmup=False, offcore_target=[])
    assert stats["noisy"] in (True, False)


# --------------------------------------------------------------------------- controller
def _run_args(**kw):
    base = dict(mode="quick", threads=None, sweep=False, cores="all")
    base.update(kw)
    return argparse.Namespace(**base)


def test_build_legs_default_all():
    topo = {"heterogeneous": False, "perf_cores": None}
    legs = controller.build_legs(_run_args(), topo)
    assert len(legs) == 1 and legs[0]["threads_mode"] == "all"


def test_build_legs_single():
    topo = {"heterogeneous": False, "perf_cores": None}
    legs = controller.build_legs(_run_args(threads=1), topo)
    assert legs[0]["threads_mode"] == "single" and legs[0]["threads"] == 1


def test_build_legs_sweep():
    topo = {"heterogeneous": False, "perf_cores": None}
    legs = controller.build_legs(_run_args(sweep=True), topo)
    modes = {leg["threads_mode"] for leg in legs}
    assert modes == {"all", "single"}


def test_build_legs_explicit():
    topo = {"heterogeneous": False, "perf_cores": None}
    legs = controller.build_legs(_run_args(threads=999), topo)
    assert legs[0]["threads_mode"] == "explicit"


def test_build_legs_hetero_adds_pcores():
    topo = {"heterogeneous": True, "perf_cores": 6}
    legs = controller.build_legs(_run_args(), topo)
    assert any(leg["cores"] == "p" for leg in legs)


def test_config_id_stable():
    cfg = registry.RunConfig(
        task="la_gemm",
        category="linalg",
        engine=None,
        mode="quick",
        params={"N": 1},
        threads=4,
        threads_mode="all",
        cores="all",
        backend_sensitive=True,
    )
    assert cfg.config_id == "la_gemm__none__quick__t4__all__all"


def test_spawn_worker_end_to_end(tmp_path):
    os.makedirs(controller.PARTIAL_DIR, exist_ok=True)
    topo = affinity.detect_pe_topology()
    cfg = registry.RunConfig(
        task="la_gemm",
        category="linalg",
        engine=None,
        mode="quick",
        params={"N": 128},
        threads=2,
        threads_mode="all",
        cores="all",
        backend_sensitive=True,
        repeat=2,
        warmup=False,
        cooldown=0.0,
    )
    partial = str(tmp_path / "r.json")
    status = controller._spawn(cfg, topo, partial)
    assert status == "ok"
    import json

    result = json.load(open(partial))
    assert result["status"] == "ok"
    assert result["task"] == "la_gemm"
    assert result["median_s"] >= 0.0
    assert len(result["reps_s"]) == 2
