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
uv run pytest                        # smoke + equivalence tests
uv run ruff check . && uv run ruff format .
```

Stack: `numpy scipy scikit-learn pandas polars lightgbm threadpoolctl psutil rich`. Python
**3.12** only. `uv.lock` is universal (Linux/macOS/Windows × x86_64/arm64). No `numba`/`umap`
(removed). LightGBM needs an OpenMP runtime; on macOS document `brew install libomp` fallback.

---

## Architecture (§3, §9)

`cli → controller → (per config) worker subprocess → runner (timed reps) → result file`,
then `controller` collects → `scoring` → `reporting`.

- `environment.py` CPU/RAM/OS/BLAS/P-E detection · `threading_ctl.py` per-lib thread wiring ·
  `affinity.py` P/E detect, pin/QoS, residency · `memory.py` guard + swap · `datasets.py`
  seeded generators · `registry.py` `@task` · `worker.py`/`runner.py` isolation+timing ·
  `scoring.py` · `reporting.py` · `tasks/{data_prep,linalg,factorization,clustering,models,sparse}.py`.

---

## INVARIANTS — do not violate (these make the numbers correct)

**Timing (§3)**
- Only the algorithm under test is inside `ctx.timer()`. Data generation, imports, allocation,
  frame construction, and the pre-FE sort are all **untimed**.
- Warm-up is **ON** by default (1 discarded run). Reps are a **fixed `repeat`** (default 5) —
  no adaptive logic, no per-task time budget. The watchdog exists only to kill an infinite hang.
- **No reset between reps. The timed region is READ-ONLY.** Use non-mutating ops (`np.sort`
  not `arr.sort`; `df.sort_values` not in-place), no `overwrite_a/overwrite_b`, `copy_X=True`,
  `warm_start=False`. Restore *input only*, untimed, for inherently destructive ops.
  **`dp_sort` must use a non-mutating sort** or reps 2+ time an already-sorted input.

**Threading (§2)**
- One `--threads` value is honoured by ALL libs via env vars set **before any heavy import**
  in the worker: `OMP/OPENBLAS/MKL/VECLIB_MAXIMUM/NUMEXPR_MAX_THREADS` + `POLARS_MAX_THREADS`.
  Polars defaults to *logical* cores — it MUST be pinned to the chosen count or the
  pandas-vs-Polars comparison is unfair.
- Default thread count = **physical cores** (`psutil.cpu_count(logical=False)`), not logical.
- Single-thread runs pin to **one P-core** (Linux/Win affinity; macOS high QoS).
- macOS has **no hard core-type affinity** — QoS bias only. Report `enforcement`
  (`none`/`pinned`/`biased`) + measured `offcore_residency_pct`; never report what was requested.

**Isolation & I/O (§3)**
- Each `(task,size,threads,cores)` runs in its own **subprocess**; no state leaks.
- The worker writes its result to a **file** (`results/.partial/<id>.json`), NOT stdout —
  library chatter on stdout would corrupt a parsed line. stdout/stderr are logs only.
- Output is resumable JSONL; `--resume` skips completed configs. OOM/timeout/non-zero exit →
  `status: "failed"`, never a hang.

**Determinism / data (§5, §11)**
- All data synthetic via `numpy.random.default_rng(1337)` (NOT sklearn `make_*`). Estimators
  get `random_state=1337`.
- **PIN DTYPES explicitly** (`int64`, `float64`; one-hot output `uint8`). Never the
  platform-default int (`int32` on Windows, `int64` elsewhere) — that breaks cross-OS identity.
- The source frame/panel is generated **once**; pandas and Polars build from the same arrays.

**Memory (§3)** — pre-task guard skips (`status: "skipped_memory"`) if estimate > available −
2 GB reserve; swap during a task → `swapped: true`, excluded from scoring.

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
- `compare` refuses on mismatched `benchmark_version`, `reference_version`, `mode`, or per-task
  `checksum` (different work ≠ CPU difference).

---

## DEFERRED — do not fake these

- **`baselines/reference.json` does not exist yet.** Build the scoring code, but until a baseline
  is produced on the chosen reference SKU, the report shows **raw times + intra-run ratios and
  OMITS the headline**. Do NOT invent a baseline or hardcode reference numbers.
- Sizes and `mem_estimate` for the FE panel, `md_gpr`, `md_random_forest_predict`, and
  `sp_feature_hasher` are first guesses. Do NOT "tune" them without real measurements;
  `mem_estimate` for iterative/materialization-heavy tasks should be calibrated from observed
  `peak_rss_mb`, not derived analytically.

---

## Build order (§14)

1. `pyproject.toml` + `uv.lock`, `environment.py`, `cpubench env`.
2. `threading_ctl.py` + `affinity.py`.
3. `registry.py` + `datasets.py` (pinned dtypes; shared frame/panel feeds both engines).
4. `worker.py` + `runner.py` (isolation, read-only reps, file result protocol) + `memory.py`,
   then `controller.py` (spawn, watchdog, collect, `--resume`).
5. `tasks/` — one category at a time, each with a quick-mode smoke test asserting shape/dtype
   and (dual-engine tasks) a pandas≈Polars equivalence check.
6. `scoring.py` (fixed category order, e_core_delta, baseline load+extract) + `reporting.py`
   (txt first, then rich/md/html).

Each layer is testable before the next. Commit in small, reviewable steps.

---

## Definition of done (for first handoff)

`uv run cpubench run --mode quick` completes on the host OS, streams resumable per-config JSON,
emits a valid schema-v1 results file and the §7.3 txt report (raw-times mode, no headline), and
the smoke + equivalence test suite passes. Reference run and live scores follow once the SKU is
chosen.

---

## Conventions

- Tasks register via `@task(name, category, sizes={quick,normal}, modes, backend_sensitive,
  mem_estimate, engines=...)` and return a cheap correctness **checksum** (tolerant invariant;
  exact only for deterministic ops like the hashers). See the pattern in §9.
- `ctx` provides `ctx.data`, `ctx.timer()`, `ctx.params`, `ctx.threads`, `ctx.rng`.
- Long-runners omit `"quick"` from `modes`.
- Prefer editing existing modules over adding new ones; keep `tasks/` modules independent.