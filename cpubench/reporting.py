"""Reports: §7.3 plain-text (canonical) first, then rich/md/html.

The txt report is pure ASCII, ≤90 columns, deterministic ordering (registry order, fixed
category order, fixed rounding) so two pasted reports ``diff`` cleanly. Every per-task row
shows the timing (median/min) right beside the score; when no baseline exists it runs in
raw-times mode: the ``score`` column is dropped and the headline is omitted.
"""

from __future__ import annotations

from cpubench import CATEGORY_LABELS, CATEGORY_ORDER, registry

WIDTH = 90
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


def _rss(x: float | None) -> str:
    if x is None:
        return "-"
    return f"{x:.0f}"


def _leg_tag(r: dict) -> str:
    """Human-readable leg name for a result (mirrors controller._leg_name)."""
    tm, cores = r.get("threads_mode"), r.get("cores")
    if tm == "single":
        return "1-core"
    if tm == "explicit":
        return f"{r.get('threads')}t"
    if cores == "p":
        return "P-cores"
    if cores == "e":
        return "E-cores"
    return "all-cores"


def _in_bucket(r: dict, bucket: str | None) -> bool:
    """Whether a result belongs to the named scoring bucket (mirrors scoring.bucket_of)."""
    tm, cores = r.get("threads_mode"), r.get("cores")
    if bucket == "all_cores":
        return tm == "all" and cores == "all"
    if bucket == "single_core":
        return tm == "single"
    if bucket == "p_cores":
        return tm == "all" and cores == "p"
    if bucket == "e_cores":
        return tm == "all" and cores == "e"
    return True


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

    # --sweep: show the single-core column beside all-cores (per-core vs whole-machine).
    single = scores.get("single_core", {})
    swept = single.get("headline") is not None
    by_cat = entry.get("by_category", {})
    single_cat = single.get("by_category", {})

    if swept:
        lines.append(f"   {'':<31}{'all':>5}{'1-core':>8}")
        lines.append(f"   {'OVERALL':<31}{round(headline):>5}{round(single['headline']):>8}")
        lines.append("   " + "." * 31)
        for cat in CATEGORY_ORDER:
            if cat in by_cat:
                sc_one = round(single_cat[cat] * 100) if cat in single_cat else "-"
                lines.append(
                    f"   {CATEGORY_LABELS[cat]:<31}{round(by_cat[cat] * 100):>5}{sc_one:>8}"
                )
        scaling = headline / single["headline"] if single["headline"] else 0.0
        lines.append("   " + "." * 31)
        lines.append(f"   {'scaling (all / 1-core)':<31}{scaling:>5.2f}")
        return lines

    lines.append(f"   {'OVERALL  (all cores)':<34}{round(headline):>5}")
    lines.append("   " + "." * 34)
    for cat in CATEGORY_ORDER:
        if cat in by_cat:
            lines.append(f"   {CATEGORY_LABELS[cat]:<34}{round(by_cat[cat] * 100):>5}")
    return lines


def _e_core_block(document: dict, scores: dict) -> list[str]:
    """Heterogeneous-only E-CORE CONTRIBUTION block (omitted on homogeneous chips)."""
    delta = scores.get("e_core_delta")
    if not delta:
        return []
    lines = [MINOR, " E-CORE CONTRIBUTION            e_core_delta  (>0 => E-cores help)"]
    p_entry = scores.get("p_cores", {})
    p_headline = p_entry.get("headline")
    if p_headline is not None:
        lines.append(f"   p-cores overall:   {round(p_headline)}")
    swept = scores.get("single_core", {}).get("headline") is not None
    if not swept:
        lines.append("   (single-P-core leg absent; pass --sweep for the full triplet)")
    for _cat, name in _ordered_entries():
        if name in delta:
            lines.append(f"   {name:<40}{delta[name]:>8.2f}")
    return lines


def _per_task(document: dict, scores: dict, bucket: str | None) -> list[str]:
    raw_mode = not scores.get("reference_present")
    entry = scores.get(bucket, {}) if bucket else {}
    per_task = entry.get("per_task", {}) if not raw_mode else {}
    # Show the primary (headline) leg's timings — not whichever leg happened to sort last.
    all_results = document.get("results", [])
    primary = [r for r in all_results if _in_bucket(r, bucket)] if bucket else []
    results = {r["task"]: r for r in (primary or all_results)}

    scored = bool(per_task) and not raw_mode

    # Column suffix (right-aligned), identical widths for the header titles and the data rows
    # so scores and timings line up. Timings (median + min) sit beside the score.
    def _cols(score, median, mn, cv, rss) -> str:
        out = f"{score:>7}" if scored else ""
        return out + f"{median:>11}{mn:>11}{cv:>7}{rss:>9}"

    name_w = 27
    head = " PER-TASK".ljust(3 + name_w)
    head += _cols("score", "median(s)", "min(s)", "cv", "rss(MB)")
    lines = [MINOR, head, MINOR]

    last_cat = None
    for cat, name in _ordered_entries():
        r = results.get(name)
        if r is None:
            continue
        if cat != last_cat:
            lines.append(f" {CATEGORY_LABELS[cat]}")
            last_cat = cat
        prefix = f"   {name:<{name_w}}"
        if r.get("status") != "ok":
            lines.append(prefix + (f"{'-':>7}" if scored else "") + f"{'FAILED':>11}")
            continue
        sc = _2dp(per_task.get(name)) if scored else ""
        lines.append(
            prefix
            + _cols(sc, _sig3(r.get("median_s")), _sig3(r.get("min_s")),
                    _2dp(r.get("cv")), _rss(r.get("peak_rss_mb")))
        )
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
    # On heterogeneous chips a task has >1 leg; annotate each note entry with its leg and
    # dedup so a per-leg result doesn't appear twice.
    multi = len({(r.get("threads_mode"), r.get("cores")) for r in results}) > 1

    def _lab(r: dict) -> str:
        return f"{r['task']} ({_leg_tag(r)})" if multi else r["task"]

    def _dedup(seq) -> list[str]:
        return list(dict.fromkeys(seq))

    noisy = _dedup(
        _lab(r) for r in results if r.get("status") == "ok" and not r.get("swapped")
        and r.get("noisy")
    )
    failed = _dedup(_lab(r) for r in results if r.get("status") != "ok")
    swapped = _dedup(
        _lab(r) for r in results if r.get("status") == "ok" and r.get("swapped")
    )
    lines = [MINOR, " NOTES"]
    lines += _wrap_list("noisy (cv>0.10):    ", noisy, "(none)")
    lines += _wrap_list("failed:             ", failed, "(none)")
    lines += _wrap_list("swapped (excluded): ", swapped, "(none)")
    entry = scores.get(bucket, {}) if bucket else {}
    empty = entry.get("empty_categories")
    if empty:
        lines += _wrap_list("empty categories:   ", empty, "(none)")
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
    lines += _e_core_block(document, scores)
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


def print_rich(document: dict, console=None) -> None:
    """Coloured console table (SPEC §7.2): per-task median/cv/RSS grouped by category.

    Noisy rows are marked, pandas vs Polars sit on adjacent rows (engine in the name), and the
    canonical copy-paste artifact remains the §7.3 txt report. Falls back to txt if rich is
    unavailable.
    """
    try:
        from rich.console import Console
        from rich.table import Table
    except Exception:  # pragma: no cover - rich is a core dep, defensive only
        print(render_txt(document))
        return

    console = console or Console()
    scores = document.get("scores", {})
    bucket = _primary_bucket(scores)
    scored = scores.get("reference_present") and (scores.get(bucket, {}) or {}).get("per_task")
    per_task = (scores.get(bucket, {}) or {}).get("per_task", {}) if scored else {}
    results = {r["task"]: r for r in document.get("results", [])}

    env = document.get("environment", {})
    cfg = document.get("config", {})
    overall = (
        str(round(scores[bucket]["headline"]))
        if scored and scores[bucket].get("headline") is not None
        else "raw times (no baseline)"
    )
    bench = document.get("benchmark_version", "?")
    console.rule(f"[bold]ml-cpu-bench {bench}[/]  OVERALL {overall}")
    console.print(
        f"{env.get('cpu_model')} ({env.get('arch')})  |  "
        f"{cfg.get('threads')} threads / cores={cfg.get('cores')}  |  "
        f"BLAS {env.get('blas_backend')}"
    )

    table = Table(show_lines=False, header_style="bold")
    table.add_column("task")
    if scored:
        table.add_column("score", justify="right")
    table.add_column("median(s)", justify="right")
    table.add_column("cv", justify="right")
    table.add_column("RSS(MB)", justify="right")

    last_cat = None
    for cat, name in _ordered_entries():
        r = results.get(name)
        if r is None:
            continue
        if cat != last_cat:
            span = 5 if scored else 4
            table.add_row(*([f"[bold cyan]{CATEGORY_LABELS[cat]}[/]"] + [""] * (span - 1)))
            last_cat = cat
        if r.get("status") != "ok":
            cells = [f"[dim]{name}[/]", "[red]FAILED[/]"]
            cells += [""] * ((5 if scored else 4) - len(cells))
            table.add_row(*cells)
            continue
        style = "yellow" if r.get("noisy") else None
        row = [f"  {name}"]
        if scored:
            row.append(_2dp(per_task.get(name)))
        row += [_sig3(r.get("median_s")), _2dp(r.get("cv")), _sig3(r.get("peak_rss_mb"))]
        table.add_row(*row, style=style)

    console.print(table)
