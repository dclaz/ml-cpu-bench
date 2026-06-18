"""Command-line interface: ``run | list | report | compare | env``.

Handlers use lazy imports so that ``cpubench env`` (and argument parsing) work before the
heavier layers are present, and so a worker subprocess never imports the controller.
"""

from __future__ import annotations

import argparse
import sys


def _add_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--mode", choices=["quick", "normal"], default="normal")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--threads", type=int, default=None, help="thread count (default: physical)")
    g.add_argument("--sweep", action="store_true", help="run both 1-core and all-cores")
    p.add_argument("--cores", choices=["all", "p", "e"], default="all")
    p.add_argument("--tasks", default=None, help="comma-separated task names to include")
    p.add_argument("--exclude", default=None, help="comma-separated task names to exclude")
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--cooldown", type=float, default=2.0)
    p.add_argument("--timeout", type=float, default=3600.0, help="per-config hang ceiling (0=off)")
    p.add_argument("--no-warmup", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--out", default=None)
    p.add_argument("--format", choices=["txt", "md", "html", "csv"], default="txt")
    p.add_argument("--summary", action="store_true")
    p.add_argument("--no-report", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cpubench", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run the benchmark")
    _add_run_args(p_run)

    sub.add_parser("list", help="list tasks + categories + sizes")

    p_report = sub.add_parser("report", help="(re)render a report from a results JSON")
    p_report.add_argument("file")
    p_report.add_argument("--format", choices=["txt", "md", "html", "csv"], default="txt")
    p_report.add_argument("--summary", action="store_true")

    p_cmp = sub.add_parser("compare", help="diff two runs")
    p_cmp.add_argument("a")
    p_cmp.add_argument("b")

    p_bake = sub.add_parser(
        "bake-baseline",
        help="extract baselines/reference.json from a --sweep results JSON",
    )
    p_bake.add_argument("file", help="a results JSON produced with --sweep")
    p_bake.add_argument(
        "--reference-version", required=True, help="identity stamp, e.g. dev-2026.06 or ref-2026.06"
    )
    p_bake.add_argument(
        "--out", default=None, help="output path (default: baselines/reference.json)"
    )
    p_bake.add_argument(
        "--force", action="store_true", help="overwrite an existing baseline file"
    )

    # internal: run a single config in this process (worker entry point).
    p_worker = sub.add_parser("_worker", help=argparse.SUPPRESS)
    p_worker.add_argument("--config-file", required=True)

    sub.add_parser("env", help="print detected environment + backend")
    return parser


def cmd_env(_args: argparse.Namespace) -> int:
    from cpubench import affinity, environment

    pe = affinity.detect_pe_topology()
    env = environment.detect_environment(perf_cores=pe["perf_cores"], eff_cores=pe["eff_cores"])
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="cpubench environment", show_header=False)
    cores = f"{env['physical_cores']} physical / {env['logical_cores']} logical"
    if env["perf_cores"] is not None:
        cores += f"  ({env['perf_cores']}P + {env['eff_cores']}E)"
    else:
        cores += "  (homogeneous)"
    rows = [
        ("CPU", f"{env['cpu_model']} ({env['arch']})"),
        ("Cores", cores),
        ("RAM", f"{env['ram_gb']} GB"),
        ("OS", env["os"]),
        ("Python", env["python"]),
        ("BLAS", f"{env['blas_backend']} ({env['blas_threads_detected']} threads detected)"),
        ("Load", f"{env['baseline_load_pct']}%"),
        ("P/E enforcement", pe["enforcement_capability"]),
    ]
    for k, v in rows:
        table.add_row(k, str(v))
    for lib, ver in env["libs"].items():
        table.add_row(f"lib:{lib}", str(ver))
    console.print(table)
    return 0


def cmd_list(_args: argparse.Namespace) -> int:
    from cpubench import registry
    from cpubench.reporting import render_task_list

    registry.load_all_tasks()
    print(render_task_list())
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from cpubench.controller import run_benchmark

    return run_benchmark(args)


def cmd_report(args: argparse.Namespace) -> int:
    from cpubench.controller import rerender_report

    return rerender_report(args.file, fmt=args.format, summary=args.summary)


def cmd_compare(args: argparse.Namespace) -> int:
    from cpubench.scoring import compare_runs

    return compare_runs(args.a, args.b)


def cmd_bake_baseline(args: argparse.Namespace) -> int:
    import json
    import os

    from cpubench import scoring

    with open(args.file) as fh:
        document = json.load(fh)
    payload = scoring.extract_baseline(document, reference_version=args.reference_version)
    n_all = len(payload["baselines"].get("all_cores", {}))
    n_single = len(payload["baselines"].get("single_core", {}))
    if not n_single:
        print(
            "WARNING: no single_core medians found — run the suite with --sweep to capture "
            "both buckets before baking a baseline."
        )
    out = args.out or scoring.BASELINE_PATH
    if os.path.exists(out) and not args.force:
        print(f"REFUSING: {out} exists (use --force to overwrite).")
        return 2
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(
        f"Wrote {out}  (reference_version={args.reference_version}, "
        f"all_cores={n_all} tasks, single_core={n_single} tasks)"
    )
    return 0


def cmd_worker(args: argparse.Namespace) -> int:
    from cpubench.worker import worker_main

    return worker_main(args.config_file)


_DISPATCH = {
    "env": cmd_env,
    "list": cmd_list,
    "run": cmd_run,
    "report": cmd_report,
    "compare": cmd_compare,
    "bake-baseline": cmd_bake_baseline,
    "_worker": cmd_worker,
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    rc = _DISPATCH[args.command](args)
    return rc or 0


if __name__ == "__main__":
    sys.exit(main())
