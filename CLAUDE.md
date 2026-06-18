# CLAUDE.md — ml-cpu-bench

A reproducible benchmark measuring **CPU** performance on classical ML + data-science
workloads (no neural nets, no GPU). Clone, run one command, get per-task and category scores.

**`SPEC.md` is the authoritative source of truth. Read the relevant section before
implementing any module.** This file is the quick reference and the list of invariants you
must not violate. Section markers below (§N) point into `SPEC.md`.

---

## Commands

```bash
uv sync                              # install pinned deps (Python 3.12)
uv run cpubench run --mode quick     # fast subset — the dev/CI loop
uv run cpubench run                  # full normal-mode run
uv run cpubench run --sweep          # add single-core pass (per-core scores)
uv run cpubench list                 # tasks + categories + sizes
uv run cpubench env                  # detected CPU/RAM/BLAS/P-E only
uv run cpubench report FILE          # re-render a report from results JSON
uv run cpubench compare A B          # diff two runs (noise/version aware)
uv run pytest                        # shape/dtype smoke tests
uv run ruff check . && uv run ruff format .
```

Stack: `numpy scipy scikit-learn pandas polars lightgbm statsforecast threadpoolctl psutil
rich`. Python **3.12** only. `uv.lock` is universal (Linux/macOS/Windows × x86_64/arm64). No
`umap`. **`numba` is back** as a transitive dep of `statsforecast` (the `md_autoarima` task):
it JIT-compiles on first call, so the default warm-up rep absorbs the compile, and
`NUMBA_NUM_THREADS` is pinned alongside the other thread-env vars. LightGBM needs an OpenMP
runtime; on macOS document `brew install libomp` fallback.

---

## Architecture (§3, §9)

`cli → controller → (per config) worker subprocess → runner (timed reps) → result file`,
then `controller` collects → `scoring` → `reporting`.

- `environment.py` CPU/RAM/OS/BLAS/P-E detection · `threading_ctl.py` per-lib thread wiring ·
  `affinity.py` P/E detect, modular per-platform pin/QoS, residency · `memory.py` peak-RSS + swap · `datasets.py`
  seeded generators · `registry.py` `@task` · `worker.py`/`runner.py` isolation+timing ·
  `scoring.py` · `reporting.py` · `tasks/{data_prep,linalg,factorization,clustering,models,sparse}.py`.

---

## INVARIANTS — do not violate (these make the numbers correct)

**Timing (§3)**
- Only the algorithm under test is inside `ctx.timer()`. Data generation, imports, allocation,
  frame construction, and the pre-FE sort are all **untimed**.
- Warm-up is **ON** by default (1 discarded run). Reps are a **fixed `repeat`** (default 5) —
  no adaptive logic, no per-task time budget. Only time limit is a per-config **`--timeout`
  (default 3600 s = 1 h, `0` disables)** hang ceiling; timeout → `status: "failed"`.
- **No reset between reps. The timed region is READ-ONLY.** Use non-mutating ops (`np.sort`
  not `arr.sort`; `df.sort_values` not in-place), no `overwrite_a/overwrite_b`, `copy_X=True`,
  `warm_start=False`. Restore *input only*, untimed, for inherently destructive ops.
  **Sort/rank ops (e.g. `fe_rank`) must be non-mutating** or reps 2+ time an already-sorted input.

**Threading (§2)**
- One `--threads` value is honoured by ALL libs via env vars set **before any heavy import**
  in the worker: `OMP/OPENBLAS/MKL/VECLIB_MAXIMUM/NUMEXPR_MAX_THREADS` + `POLARS_MAX_THREADS`.
  Polars defaults to *logical* cores — it MUST be pinned to the chosen count or the
  pandas-vs-Polars comparison is unfair.
- Default thread count = **physical cores** (`psutil.cpu_count(logical=False)`), not logical.
- Single-thread runs pin to **one P-core** (Linux/Win affinity; macOS high QoS).
- macOS has **no hard core-type affinity** — QoS bias only (set via `taskpolicy(8)` at
  worker-spawn; `-b` for `--cores e`). Report `enforcement` (`none`/`pinned`/`biased`) +
  measured `offcore_residency_pct` (10 Hz sampling; `biased` + `>10%` ⇒ flagged leaky); never
  report what was requested.
- **P/E detection is best-effort: uncertain/unavailable ⇒ treat as homogeneous** (no guessed
  split), heterogeneous-only blocks omitted.

**Isolation & I/O (§3)**
- Each `(task,size,threads,cores)` runs in its own **subprocess**; no state leaks.
- The worker writes its result to a **file** (`results/.partial/<id>.json`), NOT stdout —
  library chatter on stdout would corrupt a parsed line. stdout/stderr are logs only.
- Resume is driven by the **per-config partial files** (`results/.partial/<id>.json`) — they
  ARE the incremental stream and the sole resume source of truth (no separate JSONL log);
  `--resume` skips configs whose partial file exists, and the final schema-v1 JSON is assembled
  from them at end-of-run. OOM / timeout / non-zero exit → `status: "failed"` (no partial file,
  so it's re-attempted). (No `skipped_memory` status — that guard is gone.)

**Determinism / data (§5, §11)**
- All data synthetic via `numpy.random.default_rng(1337)` (NOT sklearn `make_*`). Estimators
  get `random_state=1337`.
- **PIN DTYPES explicitly** (`int64`, `float64`; one-hot output `uint8`). Never the
  platform-default int (`int32` on Windows, `int64` elsewhere) — that breaks cross-OS identity.
- The source frame/panel is generated **once**; pandas and Polars build from the same arrays.
  Both engines run, but their outputs are **not** asserted equal (speed, not agreement).

**Memory (§3)** — **no pre-task estimate guard and no `mem_estimate`.** Record `peak_rss_mb` per
config and tune sizes down to target machines from it; a too-big task OOM-fails. Swap during a
task → `swapped: true`, excluded from scoring.

**Checksums (§3, §11)** — **informational only**: no tolerance calibration, not a `compare`
gate. Correctness is guarded by the shape/dtype smoke suite.

---

## Scoring (§8) — get this exactly right

- Per-task score = `reference_median_s / measured_median_s` (>1 = faster than reference).
- **pandas and Polars are SEPARATE scored entries** (`task[pandas]`, `task[polars]`); both feed
  the `data_prep` category geomean. No blended data-prep number.
- Category score = geomean of its per-task scores. **Headline = 100 × geomean of the 6 category
  scores** (category-weighted, so task count doesn't skew it). Reference machine = 100.
- **Fixed category order:** `data_prep, linalg, factorization, clustering, models, sparse`.
- `all_cores` is the **headline**; `single_core` (under `--sweep`) is the secondary per-core
  diagnostic. `p_cores` and `e_core_delta` only on heterogeneous chips. **Never blend** these.
- Exclude `status != "ok"` or `swapped`. Empty category drops from the headline (and the report
  says so). Guard the geomean against zero/negative scores.
- `backend_sensitive` is a **per-task tag** (linalg, factorization, `md_gpr`), not category-derived.

---

## Output (§7)

- Authoritative JSON: `schema_version: 1`, `benchmark_version: "1.1.0"`.
- Default report is the **plain-text §7.3 format**: ASCII, ≤72 cols, deterministic ordering,
  scores as integers, leads with OVERALL then the 6-category block. Built for copy-paste diffing.
- `compare` refuses on mismatched `benchmark_version`, `reference_version`, or `mode`. A checksum
  difference is noted informationally, **not** a refusal. Non-standard `--threads N` (neither 1
  nor physical-all) → raw times only, no scored bucket, labelled as such.

---

## DEFERRED — do not fake these

- **`baselines/reference.json` does not exist yet.** Until it does, the report shows **raw times
  + intra-run ratios and OMITS the headline**. Do NOT invent a baseline or hardcode reference
  numbers. *For now* the reference is a **dev-machine placeholder** (produced on our own machine
  to exercise scoring); a publicly-rentable **cloud SKU** baseline replaces it (a
  `reference_version` bump) before release.
- Sizes for the FE panel, `md_gpr`, `md_rf_predict`, and `sp_fhash` are first guesses. Do NOT
  "tune" them without real measurements — tune sizes down from observed `peak_rss_mb`/time on
  first runs ("measure then tune"). There is no `mem_estimate` to maintain.

---

## Build order (§14)

1. `pyproject.toml` + `uv.lock`, `environment.py`, `cpubench env`.
2. `threading_ctl.py` + `affinity.py`.
3. `registry.py` + `datasets.py` (pinned dtypes; shared frame/panel feeds both engines).
4. `worker.py` + `runner.py` (isolation, read-only reps, file result protocol, peak-RSS/swap)
   + `memory.py`, then `controller.py` (spawn, `--timeout` kill, collect, `--resume`).
5. `tasks/` — one category at a time, each with a quick-mode smoke test asserting shape/dtype
   (dual-engine tasks run on both engines; no cross-engine equivalence assertion).
6. `scoring.py` (fixed category order, e_core_delta, baseline load+extract) + `reporting.py`
   (txt first, then rich/md/html).

Each layer is testable before the next. Commit in small, reviewable steps.

---

## Definition of done (for first handoff)

`uv run cpubench run --mode quick` completes on the host OS, streams resumable per-config JSON,
emits a valid schema-v1 results file and the §7.3 txt report (raw-times mode, no headline), and
the shape/dtype smoke suite passes. The headline lights up once the dev-machine placeholder
baseline is committed, then is re-based on the cloud SKU before release.

---

## Conventions

- Tasks register via `@task(name, category, sizes={quick,normal}, modes, backend_sensitive,
  engines=...)` and may return a cheap **checksum** (informational only — not validated, not a
  `compare` gate). The **registry is the single source of truth** for tasks/sizes/modes (no
  `configs/*.yaml`). See the pattern in §9.
- `ctx` provides `ctx.data`, `ctx.timer()`, `ctx.params`, `ctx.threads`, `ctx.rng`.
- Long-runners omit `"quick"` from `modes`.
- Prefer editing existing modules over adding new ones; keep `tasks/` modules independent.