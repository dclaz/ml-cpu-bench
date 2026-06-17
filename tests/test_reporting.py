"""Report snapshot/format tests: width<=90, deterministic, raw vs scored modes."""

from __future__ import annotations

from cpubench import reporting

_ENV = {
    "cpu_model": "Test CPU",
    "arch": "x86_64",
    "physical_cores": 4,
    "logical_cores": 4,
    "perf_cores": None,
    "eff_cores": None,
    "ram_gb": 8.0,
    "os": "Linux test",
    "python": "3.12.3",
    "blas_backend": "OpenBLAS",
}
_CFG = {
    "mode": "quick",
    "threads_mode": "all",
    "threads": 4,
    "cores": "all",
    "seed": 1337,
    "repeat": 5,
    "warmup": True,
}


def _raw_doc():
    return {
        "benchmark_version": "1.1.0",
        "reference_version": None,
        "run_id": "testrun",
        "environment": _ENV,
        "config": _CFG,
        "results": [
            {
                "task": "la_gemm",
                "category": "linalg",
                "status": "ok",
                "median_s": 1.234,
                "cv": 0.01,
                "noisy": False,
                "swapped": False,
                "enforcement": "none",
            },
            {
                "task": "cl_gmm",
                "category": "clustering",
                "status": "failed",
                "median_s": None,
                "cv": None,
                "noisy": False,
                "swapped": False,
                "enforcement": "none",
            },
        ],
        "scores": {"reference_present": False, "all_cores": {"raw_medians": {"la_gemm": 1.234}}},
    }


def test_raw_report_width_and_content():
    txt = reporting.render_txt(_raw_doc())
    for line in txt.splitlines():
        assert len(line) <= reporting.WIDTH, f"line exceeds report width: {line!r}"
    assert "no baseline" in txt
    assert "la_gemm" in txt
    assert "FAILED" in txt  # failed task shown


def test_report_is_deterministic():
    assert reporting.render_txt(_raw_doc()) == reporting.render_txt(_raw_doc())


def test_scored_report_shows_integer_overall():
    doc = _raw_doc()
    doc["reference_version"] = "fixture-1"
    doc["scores"] = {
        "reference_present": True,
        "all_cores": {
            "headline": 142.4,
            "by_category": {"linalg": 1.71, "clustering": 1.28},
            "per_task": {"la_gemm": 1.55},
            "empty_categories": ["data_prep", "factorization", "models", "sparse"],
        },
    }
    txt = reporting.render_txt(doc)
    for line in txt.splitlines():
        assert len(line) <= reporting.WIDTH
    assert "142" in txt  # integer headline
    assert "Linalg" in txt and "171" in txt  # category score ×100, integer


def test_summary_omits_per_task():
    txt = reporting.render_txt(_raw_doc(), summary=True)
    assert "PER-TASK" not in txt


def test_sweep_shows_single_core_column_and_scaling():
    doc = _raw_doc()
    doc["reference_version"] = "fixture-1"
    doc["config"]["sweep"] = True
    doc["scores"] = {
        "reference_present": True,
        "all_cores": {"headline": 142.0, "by_category": {"linalg": 1.71},
                      "per_task": {"la_gemm": 1.55}, "empty_categories": []},
        "single_core": {"headline": 96.0, "by_category": {"linalg": 1.10},
                        "per_task": {"la_gemm": 1.0}, "empty_categories": []},
    }
    txt = reporting.render_txt(doc)
    for line in txt.splitlines():
        assert len(line) <= reporting.WIDTH
    assert "1-core" in txt
    assert "96" in txt  # single-core headline
    assert "scaling (all / 1-core)" in txt
    assert "1.48" in txt  # 142 / 96


def test_e_core_block_only_when_heterogeneous():
    doc = _raw_doc()
    doc["reference_version"] = "fixture-1"
    doc["scores"] = {
        "reference_present": True,
        "all_cores": {"headline": 140.0, "by_category": {"linalg": 1.4},
                      "per_task": {"la_gemm": 1.4}, "empty_categories": []},
        "p_cores": {"headline": 138.0, "by_category": {"linalg": 1.38},
                    "per_task": {"la_gemm": 1.38}},
        "e_core_delta": {"la_gemm": 0.31},
    }
    txt = reporting.render_txt(doc)
    for line in txt.splitlines():
        assert len(line) <= reporting.WIDTH
    assert "E-CORE CONTRIBUTION" in txt
    assert "0.31" in txt
    # homogeneous (no e_core_delta) → block absent
    doc["scores"].pop("e_core_delta")
    doc["scores"].pop("p_cores")
    assert "E-CORE CONTRIBUTION" not in reporting.render_txt(doc)


def test_print_rich_runs():
    # smoke: rich rendering should not raise on a raw-mode doc
    reporting.print_rich(_raw_doc())


def test_long_notes_lists_wrap_under_72():
    doc = _raw_doc()
    # many noisy tasks → the NOTES list must wrap, never exceed 72 cols
    doc["results"] = [
        {"task": f"some_task_with_longish_name_{i}", "category": "linalg", "status": "ok",
         "median_s": 1.0, "cv": 0.5, "noisy": True, "swapped": False, "enforcement": "none"}
        for i in range(40)
    ]
    txt = reporting.render_txt(doc)
    for line in txt.splitlines():
        assert len(line) <= reporting.WIDTH, f"line exceeds report width: {line!r}"
    assert "noisy (cv>0.10)" in txt
