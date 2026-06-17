"""Controller: spawn workers in registry order, collect partial files, assemble schema-v1 JSON.

Each ``(task, engine, size, threads-leg)`` config runs in its own subprocess. Partial files
(`results/.partial/<config_id>.json`) ARE the resume source of truth and the incremental
stream; the authoritative results JSON is assembled from them at end of run.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime

import psutil

from cpubench import BENCHMARK_VERSION, SCHEMA_VERSION, affinity, environment, registry

RESULTS_DIR = "results"
PARTIAL_DIR = os.path.join(RESULTS_DIR, ".partial")


def _physical_cores() -> int:
    return psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 1


def build_legs(args: argparse.Namespace, topo: dict) -> list[dict]:
    """Translate CLI flags + topology into threads-legs ``{threads, threads_mode, cores}``."""
    physical = _physical_cores()
    legs: list[dict] = []

    if args.sweep:
        legs.append({"threads": physical, "threads_mode": "all", "cores": args.cores})
        legs.append({"threads": 1, "threads_mode": "single", "cores": args.cores})
    elif args.threads is None:
        legs.append({"threads": physical, "threads_mode": "all", "cores": args.cores})
    elif args.threads == 1:
        legs.append({"threads": 1, "threads_mode": "single", "cores": args.cores})
    elif args.threads == physical:
        legs.append({"threads": physical, "threads_mode": "all", "cores": args.cores})
    else:
        legs.append({"threads": args.threads, "threads_mode": "explicit", "cores": args.cores})

    # Heterogeneous auto-expansion (SPEC §2.4): add the p-cores pass for the cross-machine block.
    if topo["heterogeneous"] and args.cores == "all" and args.threads is None:
        perf = topo["perf_cores"] or physical
        legs.append({"threads": perf, "threads_mode": "all", "cores": "p"})

    return legs


def _stderr_reason(returncode: int, stderr: str) -> str:
    """Build a concise failure reason from a worker's exit code + stderr tail."""
    if returncode < 0:  # killed by signal (POSIX); -9 = SIGKILL, the OOM-killer's signature
        sig = -returncode
        hint = " (likely OOM)" if sig == 9 else ""
        return f"worker killed by signal {sig}{hint}"
    # Last non-empty stderr line is usually the exception type + message.
    tail = [ln.strip() for ln in (stderr or "").splitlines() if ln.strip()]
    detail = tail[-1] if tail else "no stderr"
    return f"worker exited {returncode}: {detail}"[:300]


def _spawn(config: registry.RunConfig, topo: dict, partial_path: str) -> tuple[str, str | None]:
    """Run one config in a subprocess. Return ``(status, reason)``; reason is None on success."""
    cfg_dict = {
        "task": config.task,
        "engine": config.engine,
        "mode": config.mode,
        "params": config.params,
        "threads": config.threads,
        "threads_mode": config.threads_mode,
        "cores": config.cores,
        "seed": config.seed,
        "repeat": config.repeat,
        "warmup": config.warmup,
        "partial_path": partial_path,
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, dir=PARTIAL_DIR) as fh:
        json.dump(cfg_dict, fh)
        cfg_path = fh.name

    prefix = affinity.taskpolicy_prefix(config.cores)
    cmd = [*prefix, sys.executable, "-m", "cpubench", "_worker", "--config-file", cfg_path]
    timeout = config.timeout if config.timeout and config.timeout > 0 else None
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            reason = _stderr_reason(proc.returncode, proc.stderr)
            sys.stderr.write(f"[worker {config.config_id}] {reason}\n{proc.stderr[-2000:]}\n")
            return "failed", reason
        if not os.path.exists(partial_path):
            reason = "worker exited 0 but wrote no result file"
            sys.stderr.write(f"[worker {config.config_id}] {reason}\n")
            return "failed", reason
        return "ok", None
    except subprocess.TimeoutExpired:
        reason = f"timeout after {timeout}s"
        sys.stderr.write(f"[worker {config.config_id}] {reason.upper()}\n")
        return "failed", reason
    finally:
        try:
            os.unlink(cfg_path)
        except OSError:
            pass


def _failed_entry(config: registry.RunConfig, reason: str) -> dict:
    return {
        "task": config.scored_name,
        "base_task": config.task,
        "engine": config.engine,
        "category": config.category,
        "backend_sensitive": config.backend_sensitive,
        "mode": config.mode,
        "threads": config.threads,
        "threads_mode": config.threads_mode,
        "cores": config.cores,
        "enforcement": "none",
        "params": config.params,
        "checksum": None,
        "import_time_s": None,
        "reps_s": [],
        "median_s": None,
        "min_s": None,
        "std_s": None,
        "cv": None,
        "noisy": False,
        "peak_rss_mb": None,
        "cpu_wall_ratio": None,
        "swapped": False,
        "offcore_residency_pct": 0.0,
        "status": "failed",
        "error": reason,
    }


def run_benchmark(args: argparse.Namespace) -> int:
    os.makedirs(PARTIAL_DIR, exist_ok=True)
    registry.load_all_tasks()

    topo = affinity.detect_pe_topology()
    env = environment.detect_environment(perf_cores=topo["perf_cores"], eff_cores=topo["eff_cores"])

    if env["baseline_load_pct"] is not None and env["baseline_load_pct"] > 25.0:
        sys.stderr.write(
            f"WARNING: machine not idle (load {env['baseline_load_pct']}%); results may be noisy.\n"
        )

    legs = build_legs(args, topo)
    threads_mode = legs[0]["threads_mode"]
    tasks = args.tasks.split(",") if args.tasks else None
    exclude = args.exclude.split(",") if args.exclude else None

    known = registry.all_task_names()
    unknown = sorted(set((tasks or []) + (exclude or [])) - known)
    if unknown:
        sys.stderr.write(
            f"WARNING: unknown task name(s) in --tasks/--exclude: {', '.join(unknown)}\n"
        )

    configs = registry.expand_configs(
        mode=args.mode,
        legs=legs,
        tasks=tasks,
        exclude=exclude,
        seed=args.seed,
        repeat=args.repeat,
        warmup=not args.no_warmup,
        timeout=args.timeout,
        cooldown=args.cooldown,
    )

    print(f"Running {len(configs)} configs (mode={args.mode}, legs={len(legs)})...")
    results: list[dict] = []
    for i, config in enumerate(configs, 1):
        partial_path = os.path.join(PARTIAL_DIR, f"{config.config_id}.json")
        label = f"[{i}/{len(configs)}] {config.config_id}"
        if args.resume and os.path.exists(partial_path):
            print(f"{label}  (resume: skip)")
            with open(partial_path) as fh:
                results.append(json.load(fh))
            continue
        # Stale partial from a prior run of the same id — overwrite.
        if os.path.exists(partial_path):
            os.unlink(partial_path)
        print(f"{label}  running...", flush=True)
        status, reason = _spawn(config, topo, partial_path)
        if status == "ok":
            with open(partial_path) as fh:
                results.append(json.load(fh))
        else:
            results.append(_failed_entry(config, reason or "worker failed"))
        if config.cooldown and i < len(configs):
            time.sleep(config.cooldown)

    # representative import time
    import_times = [r["import_time_s"] for r in results if r.get("import_time_s")]
    if import_times:
        env["import_time_s"] = round(sorted(import_times)[len(import_times) // 2], 3)

    run_id = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ") + "_" + secrets.token_hex(3)
    document = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_version": BENCHMARK_VERSION,
        "reference_version": None,
        "run_id": run_id,
        "config": {
            "mode": args.mode,
            "threads_mode": threads_mode,
            "threads": legs[0]["threads"],
            "cores": args.cores,
            "seed": args.seed,
            "repeat": args.repeat,
            "warmup": not args.no_warmup,
            "cooldown_s": args.cooldown,
            "timeout_s": args.timeout,
            "sweep": bool(args.sweep),
        },
        "environment": env,
        "results": results,
    }

    from cpubench import scoring

    document["scores"] = scoring.score_run(document)

    out_path = args.out or os.path.join(RESULTS_DIR, f"{run_id}.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(document, fh, indent=2)
    print(f"\nResults: {out_path}")

    if not args.no_report:
        from cpubench import reporting

        # Canonical artifact: the §7.3 plain-text report, always written to disk.
        report_text = reporting.render_txt(document, summary=args.summary)
        report_path = os.path.splitext(out_path)[0] + ".txt"
        with open(report_path, "w") as fh:
            fh.write(report_text)
        # Console: rich table (§7.2) for a full run; the txt summary for --summary.
        if args.summary:
            print(report_text)
        else:
            reporting.print_rich(document)
        print(f"Report: {report_path}")
        if args.format in ("md", "html"):
            extra = reporting.render_alt(document, fmt=args.format)
            alt_path = os.path.splitext(out_path)[0] + f".{args.format}"
            with open(alt_path, "w") as fh:
                fh.write(extra)
            print(f"Report ({args.format}): {alt_path}")

    return 0


def rerender_report(path: str, *, fmt: str = "txt", summary: bool = False) -> int:
    with open(path) as fh:
        document = json.load(fh)
    from cpubench import reporting

    if fmt == "txt":
        print(reporting.render_txt(document, summary=summary))
    else:
        print(reporting.render_alt(document, fmt=fmt))
    return 0
