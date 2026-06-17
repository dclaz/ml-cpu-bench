"""Reports: §7.3 plain-text (canonical) first, then rich/md/html.

The txt report is pure ASCII, ≤72 columns, deterministic ordering (registry order, fixed
category order, fixed rounding) so two pasted reports ``diff`` cleanly. When no baseline exists
it runs in raw-times mode: the ``score`` column is dropped and the headline is omitted.
"""

from __future__ import annotations

from cpubench import CATEGORY_LABELS, CATEGORY_ORDER, registry

WIDTH = 72
MAJOR = "=" * WIDTH
MINOR = "-" * WIDTH


def _sig3(x: float | None) -> str:
    if x is None:
        return "-"
    return f"{x:.3g}"


def _2dp(x: float | None) -> str:
    if x is None:
        return "-"
    return f"{x:.2f}"


def _ordered_entries() -> list[tuple[str, str]]:
    """(category, scored_name) in registry order — the canonical display order."""
    registry.load_all_tasks()
    out: list[tuple[str, str]] = []
    for spec in registry.get_registry():
        for engine in spec.engines or (None,):
            out.append((spec.category, registry.scored_name(spec.name, engine)))
    return out


def _primary_bucket(scores: dict) -> str | None:
    for bucket in ("all_cores", "single_core", "p_cores", "e_cores"):
        if bucket in scores:
            return bucket
    return None


def _header_lines(document: dict, overall: str) -> list[str]:
    env = document.get("environment", {})
    cfg = document.get("config", {})
    cores = f"{env.get('physical_cores')} physical / {env.get('logical_cores')} logical"
    if env.get("perf_cores"):
        cores += f"   ({env['perf_cores']}P + {env['eff_cores']}E)"
    else:
        cores += "   (homogeneous)"
    enforce = "none"
    for r in document.get("results", []):
        if r.get("enforcement") and r["enforcement"] != "none":
            enforce = r["enforcement"]
            break
    bench = document.get("benchmark_version", "?")
    title = f" ml-cpu-bench {bench}      OVERALL  {overall}"
    label = f"{cfg.get('threads_mode', 'all')}-cores / {cfg.get('mode', '?')}"
    title = f"{title[: WIDTH - len(label) - 1]:<{WIDTH - len(label) - 1}} {label}"
    ref = document.get("reference_version") or "(none)"
    warm = "on" if cfg.get("warmup") else "off"
    return [
        MAJOR,
        title,
        MAJOR,
        f" CPU      {env.get('cpu_model')} ({env.get('arch')})"[:WIDTH],
        f" Cores    {cores}",
        f" Threads  {cfg.get('threads')}   cores={cfg.get('cores')}   enforcement={enforce}",
        f" BLAS     {env.get('blas_backend')}",
        f" System   {env.get('os')}   Python {env.get('python')}",
        f" Run      seed {cfg.get('seed')}   repeat {cfg.get('repeat')}   "
        f"warmup {warm}   ref {ref}",
    ]


def _score_summary(document: dict, scores: dict, bucket: str | None) -> list[str]:
    lines = [MINOR, " SCORE SUMMARY                                       reference = 100", MINOR]
    raw_mode = not scores.get("reference_present")
    cfg = document.get("config", {})
    entry = scores.get(bucket, {}) if bucket else {}
    headline = entry.get("headline")

    if cfg.get("threads_mode") == "explicit":
        lines.append("   OVERALL  (threads=N, non-standard -- raw times only)")
        return lines
    if raw_mode or headline is None:
        lines.append("   OVERALL  (no baseline -- raw times only)")
        return lines

    lines.append(f"   {'OVERALL  (all cores)':<34}{round(headline):>5}")
    lines.append("   " + "." * 34)
    by_cat = entry.get("by_category", {})
    for cat in CATEGORY_ORDER:
        if cat in by_cat:
            label = CATEGORY_LABELS[cat]
            lines.append(f"   {label:<34}{round(by_cat[cat] * 100):>5}")
    return lines


def _per_task(document: dict, scores: dict, bucket: str | None) -> list[str]:
    raw_mode = not scores.get("reference_present")
    entry = scores.get(bucket, {}) if bucket else {}
    per_task = entry.get("per_task", {}) if not raw_mode else {}
    results = {r["task"]: r for r in document.get("results", [])}

    if raw_mode or not per_task:
        header = " PER-TASK                                       median(s)     cv"
        lines = [MINOR, header, MINOR]
        scored = False
    else:
        header = " PER-TASK                                  score   median(s)     cv"
        lines = [MINOR, header, MINOR]
        scored = True

    last_cat = None
    for cat, name in _ordered_entries():
        r = results.get(name)
        if r is None:
            continue
        if cat != last_cat:
            lines.append(f" {CATEGORY_LABELS[cat]}")
            last_cat = cat
        median = r.get("median_s")
        cv = r.get("cv")
        if r.get("status") != "ok":
            lines.append(f"   {name:<40}{'FAILED':>10}")
            continue
        if scored:
            sc = per_task.get(name)
            lines.append(f"   {name:<36}{_2dp(sc):>6}{_sig3(median):>11}{_2dp(cv):>8}")
        else:
            lines.append(f"   {name:<42}{_sig3(median):>10}{_2dp(cv):>8}")
    return lines


def _wrap_list(label: str, items: list[str], placeholder: str) -> list[str]:
    """Emit ``   <label>   item, item, ...`` wrapped so no line exceeds WIDTH."""
    prefix = f"   {label}"
    indent = " " * len(prefix)
    if not items:
        return [f"{prefix}{placeholder}"]
    out: list[str] = []
    cur = prefix
    for i, item in enumerate(items):
        piece = item + ("," if i < len(items) - 1 else "")
        candidate = cur + (piece if cur.endswith(" ") else " " + piece)
        if len(candidate) > WIDTH and cur.strip() != label.strip():
            out.append(cur.rstrip())
            cur = indent + piece
        else:
            cur = candidate
    out.append(cur.rstrip())
    return out


def _notes(document: dict, scores: dict, bucket: str | None) -> list[str]:
    results = document.get("results", [])
    noisy = [r["task"] for r in results if r.get("status") == "ok" and r.get("noisy")]
    excluded = [r["task"] for r in results if r.get("status") != "ok" or r.get("swapped")]
    lines = [MINOR, " NOTES"]
    lines += _wrap_list("noisy (cv>0.10):   ", noisy, "(none)")
    lines += _wrap_list("excluded:          ", excluded, "(none skipped / failed / swapped)")
    entry = scores.get(bucket, {}) if bucket else {}
    empty = entry.get("empty_categories")
    if empty:
        lines += _wrap_list("empty categories:  ", empty, "(none)")
    # p/e isolation note
    biased = [r for r in results if r.get("enforcement") == "biased"]
    if biased:
        mean_off = sum(r.get("offcore_residency_pct", 0.0) for r in biased) / len(biased)
        lines.append(
            f"   p/e isolation:     biased (QoS); mean off-core residency {mean_off * 100:.1f}%"
        )
    return lines


def render_txt(document: dict, *, summary: bool = False) -> str:
    scores = document.get("scores", {})
    bucket = _primary_bucket(scores)
    cfg = document.get("config", {})
    raw_mode = not scores.get("reference_present")
    if cfg.get("threads_mode") == "explicit":
        overall = "(threads non-standard)"
    elif raw_mode or (scores.get(bucket, {}) or {}).get("headline") is None:
        overall = "(no baseline)"
    else:
        overall = str(round(scores[bucket]["headline"]))

    lines = _header_lines(document, overall)
    lines += _score_summary(document, scores, bucket)
    if not summary:
        lines += _per_task(document, scores, bucket)
        lines += _notes(document, scores, bucket)
    lines.append(MINOR)
    lines.append(" Comparable only across runs with matching version + mode + BLAS.")
    lines.append(f" JSON: {document.get('run_id', '?')}.json")
    lines.append(MAJOR)
    return "\n".join(lines) + "\n"


def render_task_list() -> str:
    registry.load_all_tasks()
    lines = ["ml-cpu-bench tasks (registry order):"]
    last_cat = None
    for spec in registry.get_registry():
        if spec.category != last_cat:
            lines.append(f"\n[{spec.category}]  ({CATEGORY_LABELS[spec.category]})")
            last_cat = spec.category
        engines = f"  engines={spec.engines}" if spec.engines else ""
        modes = ",".join(sorted(spec.modes))
        sizes = " | ".join(f"{m}:{spec.sizes.get(m)}" for m in sorted(spec.sizes))
        bs = "  [backend_sensitive]" if spec.backend_sensitive else ""
        lines.append(f"  {spec.name:<16} modes={modes}{engines}{bs}")
        lines.append(f"      sizes: {sizes}")
    return "\n".join(lines)


def render_alt(document: dict, *, fmt: str = "md") -> str:
    """Minimal markdown/html wrapper around the canonical txt report."""
    txt = render_txt(document)
    if fmt == "md":
        return f"# ml-cpu-bench report\n\n```\n{txt}```\n"
    if fmt == "html":
        return f"<html><body><pre>\n{txt}</pre></body></html>\n"
    return txt
