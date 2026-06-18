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


def _leg(task, cat, cores, threads, median, **kw):
    return {"task": task, "category": cat, "status": kw.get("status", "ok"),
            "threads_mode": "all", "cores": cores, "threads": threads,
            "median_s": median, "min_s": (median * 0.98 if median else None),
            "cv": kw.get("cv", 0.03), "peak_rss_mb": 120.0,
            "swapped": kw.get("swapped", False), "noisy": kw.get("noisy", False),
            "enforcement": "none", "backend_sensitive": False}


def _hetero_doc(results):
    from cpubench import scoring

    doc = {
        "benchmark_version": "1.1.0", "reference_version": None, "run_id": "sim",
        "environment": {**_ENV, "perf_cores": 4, "eff_cores": 4, "physical_cores": 8,
                        "logical_cores": 8},
        "config": {"mode": "quick", "threads_mode": "all", "threads": 8, "cores": "all",
                   "seed": 1337, "repeat": 5, "warmup": True},
        "results": results,
    }
    doc["scores"] = scoring.score_run(doc, baseline_path="/does-not-exist.json")
    return doc


def test_per_task_shows_primary_all_cores_leg():
    # la_gemm runs on both legs; the all-cores leg (0.06) must be the one displayed.
    doc = _hetero_doc([
        _leg("la_gemm", "linalg", "all", 8, 0.06),
        _leg("la_gemm", "linalg", "p", 4, 0.10),
    ])
    txt = reporting.render_txt(doc)
    # the PER-TASK row carries the RSS column ("120"); the E-CORE row does not
    pt_lines = [
        line for line in txt.splitlines()
        if line.strip().startswith("la_gemm") and "120" in line
    ]
    assert len(pt_lines) == 1
    assert "0.06" in pt_lines[0]  # all-cores leg, not the 0.10 p-cores leg


def test_notes_dedup_and_split_failed_vs_swapped():
    doc = _hetero_doc([
        _leg("md_rf", "models", "all", 8, 2.46, swapped=True),
        _leg("md_rf", "models", "p", 4, 3.0, swapped=True),
        _leg("cl_gmm", "clustering", "all", 8, None, status="failed"),
        _leg("cl_gmm", "clustering", "p", 4, None, status="failed"),
    ])
    txt = reporting.render_txt(doc)
    for line in txt.splitlines():
        assert len(line) <= reporting.WIDTH
    assert "swapped (excluded):" in txt
    assert "failed:" in txt
    # leg-annotated and deduped (each leg once, not doubled into a single token)
    assert "md_rf (all-cores)" in txt and "md_rf (P-cores)" in txt
    assert txt.count("md_rf (all-cores)") == 1
    assert "cl_gmm (all-cores)" in txt


def test_csv_has_header_and_distinguishes_engines():
    import csv
    import io

    doc = _hetero_doc([
        _leg("dp_groupby[pandas]", "data_prep", "all", 8, 0.5),
        _leg("dp_groupby[polars]", "data_prep", "all", 8, 0.3),
    ])
    doc["results"][0]["engine"] = "pandas"
    doc["results"][1]["engine"] = "polars"
    for r in doc["results"]:
        r["reps_s"] = [0.5, 0.51, 0.49, 0.5, 0.5]
        r["std_s"] = 0.01

    text = reporting.render_csv(doc)
    rows = list(csv.DictReader(io.StringIO(text)))
    assert [c for c in reporting.CSV_COLUMNS] == list(rows[0].keys())
    by_engine = {row["engine"]: row for row in rows}
    assert set(by_engine) == {"pandas", "polars"}
    assert by_engine["pandas"]["task"] == "dp_groupby[pandas]"
    assert by_engine["polars"]["task"] == "dp_groupby[polars]"
    assert by_engine["pandas"]["reps"] == "5"
    assert by_engine["pandas"]["bucket"] == "all_cores"


def test_csv_blank_score_without_baseline_and_failed_rows():
    import csv
    import io

    doc = _hetero_doc([
        _leg("la_gemm", "linalg", "all", 8, 0.06),
        _leg("cl_gmm", "clustering", "all", 8, None, status="failed"),
    ])
    rows = list(csv.DictReader(io.StringIO(reporting.render_csv(doc))))
    by_task = {row["task"]: row for row in rows}
    assert by_task["la_gemm"]["score"] == ""  # raw mode → no score
    assert by_task["cl_gmm"]["status"] == "failed"
    assert by_task["cl_gmm"]["median_s"] == ""  # blank for failed


def test_print_rich_runs():
    # smoke: rich rendering should not raise on a raw-mode doc
    reporting.print_rich(_raw_doc())


def test_print_rich_shows_engine_column():
    from rich.console import Console

    doc = _hetero_doc([
        _leg("dp_groupby[pandas]", "data_prep", "all", 8, 0.5),
        _leg("dp_groupby[polars]", "data_prep", "all", 8, 0.3),
    ])
    doc["results"][0].update(engine="pandas", base_task="dp_groupby")
    doc["results"][1].update(engine="polars", base_task="dp_groupby")
    console = Console(record=True, width=100)
    reporting.print_rich(doc, console=console)
    text = console.export_text()
    assert "engine" in text
    assert "pandas" in text and "polars" in text


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
