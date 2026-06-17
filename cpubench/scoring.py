"""Scoring: ratios, category geomeans (fixed order), e_core_delta, baseline load/extract.

Per-task score = ``reference_median_s / measured_median_s``. pandas & Polars are separate scored
entries. Category score = geomean of its per-task scores. Headline = ``100 × geomean(6
categories)`` — category-weighted. No ``baselines/reference.json`` is committed; when absent,
scoring returns raw-times mode (headline omitted). See SPEC §8.
"""

from __future__ import annotations

import json
import math
import os

from cpubench import CATEGORY_ORDER

BASELINE_PATH = os.path.join("baselines", "reference.json")
# Buckets that carry a reference denominator (p_cores reuses the all_cores reference).
_SCORED_BUCKETS = ("all_cores", "single_core", "p_cores")


def geomean(values) -> float:
    """Geometric mean, guarded against zero/negative inputs (those are dropped)."""
    positive = [v for v in values if v is not None and v > 0]
    if not positive:
        return 0.0
    return math.exp(sum(math.log(v) for v in positive) / len(positive))


def bucket_of(result: dict) -> str | None:
    tm = result.get("threads_mode")
    cores = result.get("cores")
    if tm == "all" and cores == "all":
        return "all_cores"
    if tm == "single":
        return "single_core"
    if tm == "all" and cores == "p":
        return "p_cores"
    if tm == "all" and cores == "e":
        return "e_cores"
    return None  # explicit --threads N → not scored


def load_baseline(path: str | None = None) -> dict | None:
    path = path or BASELINE_PATH
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def _ok(results: list[dict]) -> list[dict]:
    return [r for r in results if r.get("status") == "ok" and not r.get("swapped")]


def _raw_medians(records: list[dict]) -> dict[str, float]:
    return {r["task"]: r["median_s"] for r in records if r.get("median_s") is not None}


def _task_category(records: list[dict]) -> dict[str, str]:
    return {r["task"]: r["category"] for r in records}


def apply_baseline(records: list[dict], baseline_medians: dict[str, float]) -> dict:
    """Score a bucket against per-task reference medians.

    Returns ``{per_task, by_category, headline, empty_categories, missing}``. ``headline`` is
    None if no category survived.
    """
    cat_of = _task_category(records)
    per_task: dict[str, float] = {}
    missing: list[str] = []
    for r in records:
        name = r["task"]
        ref = baseline_medians.get(name)
        meas = r.get("median_s")
        if ref is None:
            missing.append(name)
            continue
        if meas is None or meas <= 0:
            continue
        per_task[name] = ref / meas

    by_category: dict[str, float] = {}
    empty: list[str] = []
    for cat in CATEGORY_ORDER:
        scores = [s for name, s in per_task.items() if cat_of.get(name) == cat]
        if scores:
            by_category[cat] = geomean(scores)
        else:
            empty.append(cat)

    headline = 100.0 * geomean(by_category.values()) if by_category else None
    return {
        "per_task": per_task,
        "by_category": by_category,
        "headline": headline,
        "empty_categories": empty,
        "missing": missing,
    }


def compute_e_core_delta(by_bucket: dict[str, list[dict]]) -> dict[str, float]:
    """Per-task ``(p_median - all_median) / p_median`` (intra-run; heterogeneous only)."""
    all_med = _raw_medians(by_bucket.get("all_cores", []))
    p_med = _raw_medians(by_bucket.get("p_cores", []))
    delta: dict[str, float] = {}
    for task, p in p_med.items():
        a = all_med.get(task)
        if a is not None and p > 0:
            delta[task] = round((p - a) / p, 4)
    return delta


def score_run(document: dict, baseline_path: str | None = None) -> dict:
    """Compute the ``scores`` block for a results document (raw mode when no baseline)."""
    baseline = load_baseline(baseline_path)
    ok = _ok(document.get("results", []))

    by_bucket: dict[str, list[dict]] = {}
    for r in ok:
        b = bucket_of(r)
        if b is not None:
            by_bucket.setdefault(b, []).append(r)

    scores: dict = {
        "reference_present": baseline is not None,
        "reference_version": baseline.get("reference_version") if baseline else None,
    }

    for bucket, records in by_bucket.items():
        entry: dict = {"raw_medians": _raw_medians(records)}
        if baseline is not None and bucket in _SCORED_BUCKETS:
            # p_cores reuses the all_cores reference denominator.
            ref_key = "all_cores" if bucket == "p_cores" else bucket
            ref_medians = (baseline.get("baselines", {}) or {}).get(ref_key, {})
            entry.update(apply_baseline(records, ref_medians))
        scores[bucket] = entry

    delta = compute_e_core_delta(by_bucket)
    if delta:
        scores["e_core_delta"] = delta

    return scores


def extract_baseline(document: dict, *, reference_version: str) -> dict:
    """Build a ``reference.json`` payload from a ``--sweep`` results document (SPEC §8)."""
    ok = _ok(document.get("results", []))
    baselines: dict[str, dict[str, float]] = {}
    for r in ok:
        b = bucket_of(r)
        if b in ("all_cores", "single_core") and r.get("median_s") is not None:
            baselines.setdefault(b, {})[r["task"]] = r["median_s"]
    return {
        "reference_version": reference_version,
        "benchmark_version": document.get("benchmark_version"),
        "machine": document.get("environment", {}),
        "baselines": baselines,
    }


# --------------------------------------------------------------------------- compare
def compare_runs(path_a: str, path_b: str) -> int:
    with open(path_a) as fh:
        a = json.load(fh)
    with open(path_b) as fh:
        b = json.load(fh)

    for field in ("benchmark_version", "reference_version"):
        if a.get(field) != b.get(field):
            print(f"REFUSING: {field} mismatch ({a.get(field)!r} vs {b.get(field)!r})")
            return 2
    if a.get("config", {}).get("mode") != b.get("config", {}).get("mode"):
        print("REFUSING: mode mismatch")
        return 2

    a_res = {r["task"]: r for r in a.get("results", []) if r.get("status") == "ok"}
    b_res = {r["task"]: r for r in b.get("results", []) if r.get("status") == "ok"}
    common = [t for t in a_res if t in b_res]

    sensitive, neutral = [], []
    for t in common:
        ra, rb = a_res[t], b_res[t]
        ma, mb = ra.get("median_s"), rb.get("median_s")
        if not ma or not mb:
            continue
        ratio = ma / mb
        # noise-aware: only flag when the gap exceeds combined run-to-run variance
        cva = ra.get("cv") or 0.0
        cvb = rb.get("cv") or 0.0
        threshold = 1.0 + (cva + cvb)
        flagged = ratio > threshold or ratio < 1.0 / threshold
        line = f"  {t:32s} A={ma:.4g}s  B={mb:.4g}s  ratio={ratio:.2f}"
        if flagged:
            line += "  *"
        (sensitive if ra.get("backend_sensitive") else neutral).append(line)
        if ra.get("checksum") and rb.get("checksum") and ra["checksum"] != rb["checksum"]:
            print(f"  note: checksum differs for {t} (informational)")

    print("BACKEND-SENSITIVE (linalg/factorization/md_gpr):")
    print("\n".join(sensitive) or "  (none)")
    print("BACKEND-NEUTRAL:")
    print("\n".join(neutral) or "  (none)")
    print("\n(* = gap exceeds combined run-to-run variance)")
    return 0
