"""Scoring math against a small in-repo fixture baseline (SPEC §8)."""

from __future__ import annotations

import json

import pytest

from cpubench import scoring


def _ok(task, cat, median, tm="all", cores="all"):
    return {
        "task": task,
        "category": cat,
        "status": "ok",
        "median_s": median,
        "threads_mode": tm,
        "cores": cores,
        "swapped": False,
        "backend_sensitive": False,
    }


FIXTURE_RESULTS = [
    _ok("la_gemm", "linalg", 2.0),
    _ok("la_solve", "linalg", 4.0),
    _ok("cl_kmeans", "clustering", 1.0),
    {
        "task": "cl_gmm",
        "category": "clustering",
        "status": "failed",
        "median_s": None,
        "threads_mode": "all",
        "cores": "all",
        "swapped": False,
    },
]
FIXTURE_BASELINE = {
    "reference_version": "fixture-1",
    "mode": "normal",
    "baselines": {"all_cores": {"la_gemm": 4.0, "la_solve": 4.0, "cl_kmeans": 2.0}},
}


def test_geomean_zero_and_negative_guarded():
    assert scoring.geomean([0.0, 4.0]) == pytest.approx(4.0)
    assert scoring.geomean([-1.0, 9.0, 1.0]) == pytest.approx(3.0)
    assert scoring.geomean([]) == 0.0


def test_bucket_of():
    assert scoring.bucket_of(_ok("t", "c", 1.0)) == "all_cores"
    assert scoring.bucket_of(_ok("t", "c", 1.0, tm="single")) == "single_core"
    assert scoring.bucket_of(_ok("t", "c", 1.0, cores="p")) == "p_cores"
    assert scoring.bucket_of({"threads_mode": "explicit", "cores": "all"}) is None


def test_apply_baseline_ratios_and_geomean():
    out = scoring.apply_baseline(
        [r for r in FIXTURE_RESULTS if r["status"] == "ok"],
        FIXTURE_BASELINE["baselines"]["all_cores"],
    )
    assert out["per_task"]["la_gemm"] == pytest.approx(2.0)
    assert out["per_task"]["la_solve"] == pytest.approx(1.0)
    assert out["by_category"]["linalg"] == pytest.approx(2.0**0.5)
    assert out["by_category"]["clustering"] == pytest.approx(2.0)
    # 4 categories empty (only linalg + clustering scored)
    assert set(out["empty_categories"]) == {"data_prep", "factorization", "models", "sparse"}
    expected = 100.0 * (out["by_category"]["linalg"] * out["by_category"]["clustering"]) ** 0.5
    assert out["headline"] == pytest.approx(expected)


def test_score_run_raw_mode_when_no_baseline(tmp_path):
    doc = {"results": FIXTURE_RESULTS}
    scores = scoring.score_run(doc, baseline_path=str(tmp_path / "missing.json"))
    assert scores["reference_present"] is False
    assert "raw_medians" in scores["all_cores"]
    assert scores["all_cores"].get("headline") is None


def test_score_run_scored_with_fixture(tmp_path):
    bpath = tmp_path / "ref.json"
    bpath.write_text(json.dumps(FIXTURE_BASELINE))
    doc = {"config": {"mode": "normal"}, "results": FIXTURE_RESULTS}
    scores = scoring.score_run(doc, baseline_path=str(bpath))
    assert scores["reference_present"] is True
    assert scores["reference_mode_mismatch"] is False
    assert scores["all_cores"]["headline"] is not None


def test_score_run_raw_mode_on_mode_mismatch(tmp_path):
    # quick run against a normal baseline → raw times only (different task sizes).
    bpath = tmp_path / "ref.json"
    bpath.write_text(json.dumps(FIXTURE_BASELINE))
    doc = {"config": {"mode": "quick"}, "results": FIXTURE_RESULTS}
    scores = scoring.score_run(doc, baseline_path=str(bpath))
    assert scores["reference_present"] is False
    assert scores["reference_mode_mismatch"] is True
    assert scores["reference_mode"] == "normal"
    assert scores["all_cores"].get("headline") is None


def test_e_core_delta():
    by_bucket = {
        "all_cores": [_ok("la_gemm", "linalg", 1.0)],
        "p_cores": [_ok("la_gemm", "linalg", 2.0, cores="p")],
    }
    delta = scoring.compute_e_core_delta(by_bucket)
    assert delta["la_gemm"] == pytest.approx(0.5)  # (2-1)/2


def test_score_run_legacy_baseline_defaults_to_normal(tmp_path):
    # A baseline without a "mode" key (pre-mode-field) is assumed normal-mode.
    bpath = tmp_path / "ref.json"
    bpath.write_text(json.dumps({k: v for k, v in FIXTURE_BASELINE.items() if k != "mode"}))
    normal = scoring.score_run(
        {"config": {"mode": "normal"}, "results": FIXTURE_RESULTS}, str(bpath)
    )
    quick = scoring.score_run({"config": {"mode": "quick"}, "results": FIXTURE_RESULTS}, str(bpath))
    assert normal["reference_present"] is True
    assert quick["reference_mode_mismatch"] is True


def test_extract_baseline_from_sweep():
    doc = {
        "benchmark_version": "1.1.0",
        "config": {"mode": "normal"},
        "results": [
            _ok("la_gemm", "linalg", 1.5),
            _ok("la_gemm", "linalg", 5.0, tm="single"),
        ],
    }
    out = scoring.extract_baseline(doc, reference_version="dev-1")
    assert out["baselines"]["all_cores"]["la_gemm"] == 1.5
    assert out["baselines"]["single_core"]["la_gemm"] == 5.0
    assert out["mode"] == "normal"
