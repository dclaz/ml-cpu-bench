"""Worker: run ONE config in a fresh subprocess (SPEC §3).

Order matters: thread env vars and affinity are applied **before any heavy import**, then only
the libs the task needs are imported (via ``load_all_tasks``), data is generated untimed, and
the timed reps run. The result is written to a **file** (never stdout); stdout/stderr are logs.
A failure exits non-zero and writes no file, so ``--resume`` re-attempts it.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback


def worker_main(config_file: str) -> int:
    with open(config_file) as fh:
        cfg = json.load(fh)

    # ---- BEFORE heavy imports: pin threads + affinity --------------------------------
    from cpubench import threading_ctl

    threading_ctl.set_thread_env(cfg["threads"])

    from cpubench import affinity

    topo = affinity.detect_pe_topology()
    aff = affinity.apply_affinity(cfg["threads_mode"], cfg["cores"], topo)

    # ---- heavy imports + data generation (untimed) -----------------------------------
    import numpy as np

    from cpubench import registry
    from cpubench.runner import RunContext, run_reps

    t_import0 = time.perf_counter()
    registry.load_all_tasks()
    spec = registry.get_task(cfg["task"])
    import_time_s = time.perf_counter() - t_import0

    seed = int(cfg["seed"])
    data_rng = np.random.default_rng(seed)
    data = spec.data(cfg["params"], data_rng, cfg.get("engine"))
    ctx = RunContext(
        data=data,
        params=cfg["params"],
        threads=cfg["threads"],
        rng=np.random.default_rng(seed),
        engine=cfg.get("engine"),
    )

    with threading_ctl.threadpool_limits(cfg["threads"]):
        stats = run_reps(
            spec.func,
            ctx,
            repeat=int(cfg["repeat"]),
            warmup=bool(cfg["warmup"]),
            offcore_target=aff["offcore_target"],
        )

    checksum = stats.pop("checksum")
    if checksum is not None:
        checksum = _checksum_str(checksum)

    result = {
        "task": registry.scored_name(spec.name, cfg.get("engine")),
        "base_task": spec.name,
        "engine": cfg.get("engine"),
        "category": spec.category,
        "backend_sensitive": spec.backend_sensitive,
        "mode": cfg["mode"],
        "threads": cfg["threads"],
        "threads_mode": cfg["threads_mode"],
        "cores": cfg["cores"],
        "enforcement": aff["enforcement"],
        "params": cfg["params"],
        "checksum": checksum,
        "import_time_s": round(import_time_s, 3),
        "status": "ok",
        "error": None,
        **stats,
    }

    partial_path = cfg["partial_path"]
    os.makedirs(os.path.dirname(partial_path), exist_ok=True)
    tmp = partial_path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(result, fh, indent=2, default=_json_default)
    os.replace(tmp, partial_path)
    return 0


def _checksum_str(value) -> str:
    if isinstance(value, dict):
        return ";".join(f"{k}={_fmt(v)}" for k, v in value.items())
    return _fmt(value)


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def _json_default(o):
    import numpy as np

    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _entry_point() -> int:  # pragma: no cover
    return worker_main(sys.argv[sys.argv.index("--config-file") + 1])


if __name__ == "__main__":  # pragma: no cover
    try:
        sys.exit(_entry_point())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
