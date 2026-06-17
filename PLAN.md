# ml-cpu-bench — Implementation Plan

## Context

The repo is greenfield: only `SPEC.md` (authoritative, 1000 lines), `CLAUDE.md` (invariants
+ quick reference), `README.md`, and `.gitignore` exist. No Python code, no `pyproject.toml`.
SPEC §13 marks every *design* decision as resolved, so this is a build-out task, not a design
task — the work is converting the spec into working modules in the §14 build order without
violating the timing/threading/determinism invariants.

**Goal of this pass:** a *full vertical slice* — all 6 architecture layers end-to-end
(harness → all task categories → scoring → reporting). The completion gate is the spec's
Definition of Done: `uv run cpubench run --mode quick` runs green across all 6 categories,
streams resumable per-config JSON, emits a schema-v1 results file and the §7.3 txt report in
**raw-times mode**, and the shape/dtype smoke suite passes.

**Baseline is deferred:** do **not** create `baselines/reference.json`. The report runs in
raw-times mode (headline omitted, `OVERALL (no baseline — raw times only)`). Scoring code is
still written and unit-tested against a small in-repo *fixture* baseline so the score/headline
path is exercised without committing a fake reference.

**Dev-machine reality (affects tuning, not design):** this box is an Intel i5-4570, **4
physical cores, no SMT, ~8 GB RAM (WSL2)** — homogeneous (no P/E split). `normal` sizes target
≤8 GB RSS *assuming >16 GB RAM*, so several `normal` tasks will OOM-fail here by design; **quick
mode is the dev loop**. P/E / heterogeneous code paths can't be exercised locally — they get
unit tests with mocked detection, and the homogeneous fallback is what runs here.

---

## Invariants to honour (from CLAUDE.md / SPEC §2,§3,§5,§8 — do not violate)

- **Timing:** only the algorithm is inside `ctx.timer()`. Data gen, imports, allocation, frame
  construction, pre-FE sort are untimed. Warm-up ON (1 discard); fixed `repeat=5`, no adaptive
  logic; `--timeout` (default 3600s, `0` disables) is a hang ceiling only → `status:"failed"`.
- **Read-only timed region, no reset between reps:** non-mutating ops (`np.sort` not
  `arr.sort`; `df.sort_values` not in-place), no `overwrite_*`, `copy_X=True`,
  `warm_start=False`. Destructive ops restore *input only*, untimed. `dp_sort` must be
  non-mutating.
- **Threading:** one `--threads` honoured by ALL libs via env vars set **before any heavy
  import** in the worker (`OMP/OPENBLAS/MKL/VECLIB_MAXIMUM/NUMEXPR_MAX_THREADS` +
  `POLARS_MAX_THREADS`). Default threads = physical cores. Single-thread pins one P-core.
- **Isolation/IO:** each `(task,size,threads,cores)` in its own subprocess; result written to a
  **file** `results/.partial/<id>.json`, never stdout. Partial files ARE the resume source.
- **Determinism:** all data via `numpy.random.default_rng(1337)` (not sklearn `make_*`);
  estimators `random_state=1337`. **Pin dtypes** (`int64`, `float64`, one-hot `uint8`); never
  platform-default int. Shared frame/panel generated once, fed to both engines (not asserted
  equal). Tasks iterate in registry order.
- **Memory:** no pre-task guard, no `mem_estimate`. Record `peak_rss_mb`; OOM → failed; swap →
  `swapped:true`, excluded from scoring.
- **Checksums:** informational only — not validated, not a compare gate.
- **Scoring:** per-task = `ref_median / measured_median`; pandas & Polars are separate scored
  entries; category = geomean; headline = `100 × geomean(6 categories)`. Fixed category order:
  `data_prep, linalg, factorization, clustering, models, sparse`.

---

## Layer 1 — Project skeleton + environment + `cpubench env`

**Files:** `pyproject.toml`, `uv.lock`, `cpubench/__init__.py`, `cpubench/environment.py`,
`cpubench/cli.py` (env subcommand only at first), `cpubench/__main__.py`.

1. `pyproject.toml`: Python `==3.12.*`; core deps (§10) `numpy scipy scikit-learn pandas polars
   lightgbm threadpoolctl psutil rich`; dev group `pytest ruff`. Console entry point
   `cpubench = "cpubench.cli:main"`; also support `python -m cpubench` via `__main__.py`.
   Configure ruff. Run `uv lock` then `uv sync`.
2. `environment.py` — pure detection, no side effects:
   - CPU model/arch, `logical_cores`/`physical_cores` via `psutil.cpu_count(logical=False/True)`.
   - RAM (`psutil.virtual_memory().total`), OS string, Python version, lib versions
     (`importlib.metadata`).
   - **BLAS backend** via `threadpoolctl.threadpool_info()` + `numpy.show_config()` fallback
     (`Accelerate`/`OpenBLAS`/`MKL`); `blas_threads_detected`. Note §10 caveat: Accelerate on
     macOS-arm64 is finickiest — keep detection in one function with clear fallbacks.
   - `baseline_load_pct` (pre-flight `psutil.cpu_percent`), representative `import_time_s` filled
     later by worker. Returns a dict matching the §7.1 `environment` block.
   - **P/E detection lives in `affinity.py` (Layer 2)**; `environment.py` consumes
     `perf_cores`/`eff_cores` from it (null when homogeneous/uncertain — best-effort, never guess).
3. `cli.py`: argparse with subcommands `run|list|report|compare|env` (stub all; implement `env`
   now → prints detected environment + backend only).

**Test/checkpoint:** `uv run cpubench env` prints correct CPU/RAM/cores/BLAS on this box
(expect 4/4 cores, homogeneous, OpenBLAS or MKL). Unit test `environment` dict shape.

---

## Layer 2 — `threading_ctl.py` + `affinity.py`

**Files:** `cpubench/threading_ctl.py`, `cpubench/affinity.py`.

1. `threading_ctl.py`:
   - `set_thread_env(n)` — sets `OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS,
     VECLIB_MAXIMUM_THREADS, NUMEXPR_MAX_THREADS, POLARS_MAX_THREADS` in `os.environ`. **Called
     by the worker before any heavy import.** Polars MUST be pinned (else it grabs logical cores).
   - `threadpool_limits(n)` context helper (belt-and-braces around BLAS regions).
   - Helpers to resolve `n_jobs` (sklearn) / `num_threads` (lgbm) from `ctx.threads`.
2. `affinity.py` — **modular per-platform backend** (detect → apply pin/QoS → verify residency),
   so the soft macOS path can be swapped later without touching runner/scoring/report:
   - **P/E detection** (best-effort): Linux sysfs `cpu_capacity`/core-type flags; Windows CPU-set
     efficiency class; macOS `sysctl hw.perflevel0/1.physicalcpu`. Uncertain/uniform ⇒
     **homogeneous** (`perf_cores`/`eff_cores = null`), heterogeneous blocks omitted. Never guess.
   - **Enforcement backends:** Linux/Windows hard pin via `psutil.cpu_affinity`
     (`sched_setaffinity`/`SetThreadAffinityMask`); macOS QoS via **`taskpolicy(8)` at spawn**
     (`-b` for `--cores e`; default QoS for `p`, capped at P-core count). Single-thread → one
     P-core (hard on Linux/Win, high QoS on macOS).
   - `enforcement` field returned: `none` (cores=all) / `pinned` / `biased`.
   - **Residency sampler:** low-priority thread, **10 Hz**, active only while `ctx.timer()` open,
     reads `psutil.cpu_times(percpu=True)`; on Linux/Win pinned to complement of test cores.
     Computes `offcore_residency_pct = busy_on_wrong_type / total_busy`. `biased` + `>10%` ⇒ leaky.

**Test/checkpoint:** unit-test detection with mocked sysfs/sysctl for hetero + homo + uncertain;
assert env vars set correctly; assert homogeneous path on this box (`--cores p/e` collapse to
`all`). macOS/Windows branches code-reviewed (not runnable locally) and unit-tested via mocks.

---

## Layer 3 — `registry.py` + `datasets.py`

**Files:** `cpubench/registry.py`, `cpubench/datasets.py`.

1. `registry.py`:
   - `@task(name, category, sizes={"quick":..,"normal":..}, modes={...}, backend_sensitive=False,
     engines=("pandas","polars")|None)` decorator → appends to an ordered list (registry IS source
     of truth; **deterministic registry order** preserved for run + report).
   - Config expansion: resolve `(task, size, threads, cores)` configs for a mode; expand
     `engines` into separate scored entries `name[engine]`; long-runners omit `"quick"`.
   - Heterogeneous auto-expansion (SPEC §2.4 note): on a hetero CPU a plain run adds the `cores="p"`
     pass (threads_mode "all"); single-P leg gated behind `--sweep`. Homogeneous ⇒ no expansion.
   - `config_id` helper → stable filename for `results/.partial/<id>.json`.
   - `list` rendering pulls straight from the registry.
2. `datasets.py` — seeded generators, **all dtypes pinned**, every generator takes
   `default_rng(seed)`:
   - Shared **core data-prep frame** (20 numeric float64, 3 categorical low/med/high card, 1
     datetime64[ns]) — built **once**, both engines build native frames from the same numpy arrays.
   - Shared **FE panel** (`entity_id` int64, `timestamp` datetime64, `target` f64, 20 numeric,
     3 categorical) sorted by `(entity_id, timestamp)` **untimed**.
   - Dense matrices (gemm/solve/cholesky-SPD/qr/svd/eigh/fft), factorization sets, clustering
     blobs, model X/y (regression/multiclass blobs for lgbm_multi), sparse CSR via
     `scipy.sparse.random(..., random_state=rng)`, Zipfian token corpus for tfidf/hashvec, and
     `"field=value"` string rows for `sp_fhash`.
   - Controlled ~1:1 join-key cardinality (no many-to-many blow-up).

**Test/checkpoint:** unit-test a generator produces pinned dtypes and identical bytes across two
calls with same seed; registry returns tasks in declaration order; `cpubench list` works.

---

## Layer 4 — `worker.py` + `runner.py` + `memory.py` + `controller.py`

**Files:** `cpubench/worker.py`, `cpubench/runner.py`, `cpubench/memory.py`,
`cpubench/controller.py`. Extend `cli.py run`.

1. `worker.py` — entry for ONE config in a fresh subprocess:
   - **First**: `set_thread_env(threads)` (pre-import), apply affinity/QoS, then import only the
     libs the task needs (targeted imports), then run. Records `import_time_s`.
   - Builds `ctx` = `{data, timer(), params, threads, rng}`; calls the task; writes result JSON to
     `results/.partial/<config_id>.json`. stdout/stderr = logs only.
2. `runner.py` — warm-up (1 discard, default on; `--no-warmup` off), exactly `repeat` timed reps
   via `perf_counter` (+ `process_time` for cpu_wall_ratio). **No reset between reps; timed region
   read-only.** Computes median/min/std/cv (cv>0.10 ⇒ `noisy`), collects reps, peak RSS, swap.
3. `memory.py` — peak-RSS sampler (psutil RSS high-water) + swap-in/out delta around timed region
   → `swapped` flag.
4. `controller.py` — for each config in registry order: spawn worker subprocess, enforce
   `--timeout` kill (→ `status:"failed"`, no partial file), `--cooldown` gap (default 2s),
   `--resume` skips configs whose partial file exists. At end, **assemble schema-v1 JSON** (§7.1)
   from all partial files. OOM/non-zero exit/timeout → `failed`. Quiet-machine pre-flight warning.

**Test/checkpoint:** a trivial dummy task runs in isolation, writes a partial file, assembles into
schema-v1 JSON; `--resume` skips it; timeout kills a sleeping worker → `failed`; peak RSS recorded.

---

## Layer 5 — `tasks/` (all 6 categories, §5)

**Files:** `cpubench/tasks/{data_prep,linalg,factorization,clustering,models,sparse}.py` +
`tasks/__init__.py` (imports all so `@task` registration fires in fixed category order).

Implement the full §5 catalogue. Each task: untimed data via `ctx.data`, single `ctx.timer()`
region, optional informational checksum. Key per-category notes:

- **data_prep.py** — core `dp_*` (groupby/join/sort/filter/string/rolling) + FE `fe_*`
  (lags/rolling/expanding/ewm/onehot/rank/datetime), **each on both engines** (`engines=`).
  FE must be **leakage-safe** (`closed="left"`, prior obs only) and build features into a **fresh
  structure each rep** (no mutation of shared panel). `dp_sort` non-mutating. one-hot → `uint8`.
- **linalg.py** — `la_gemm/solve/cholesky/qr/svd/eigh/fft`; `backend_sensitive=True`; no
  `overwrite_*`. SPD matrix for cholesky.
- **factorization.py** — `mf_tsvd/pca/nmf`; `backend_sensitive=True`.
- **clustering.py** — `cl_kmeans/mbkmeans/optics/gmm`; bounded `max_eps`+ball_tree for OPTICS.
- **models.py** — `md_linreg/ridge/lasso/logreg/bayes_ridge/gpr/rf/rf_predict/hist_gbm/lgbm/
  lgbm_multi/svc_rbf/knn`. `md_rf_predict`: forest trained **untimed**, predict timed.
  `md_gpr` `optimizer=None`, `backend_sensitive=True`, normal-only. `md_lgbm_multi` normal-only.
  `copy_X=True`, `warm_start=False` throughout; honour `n_jobs`/`num_threads` from `ctx.threads`.
- **sparse.py** — `sp_tfidf/hashvec/fhash/matmul/tsvd/nmf/saga` + `nlp_lda` + `sp_lasso_cv`
  (last two normal-only). `sp_fhash` watch materialized-input RSS.
- **Long-runners omit `"quick"` from `modes`:** `md_gpr, md_lgbm_multi, nlp_lda, sp_lasso_cv`.

**Test/checkpoint (the primary correctness guard):** `tests/` smoke suite asserts each task runs
in **quick** mode and returns expected **shape/dtype** (dual-engine tasks run both engines; **not**
asserted equal cross-engine). Build/test one category at a time.

---

## Layer 6 — `scoring.py` + `reporting.py`

**Files:** `cpubench/scoring.py`, `cpubench/reporting.py`. Extend `cli.py` (`report`, `compare`).

1. `scoring.py`:
   - Per-task `ref_median/measured_median`; pandas/Polars separate; category geomean in **fixed
     order**; headline `100 × geomean(6 categories)`. Bucket by `(threads_mode, cores)` →
     `all_cores`/`single_core`/`p_cores`; non-standard `--threads N` ⇒ no scored bucket.
   - `e_core_delta` (intra-run, no ref) on hetero only. Exclude `status!="ok"` or `swapped`; empty
     category drops from headline (report says so); guard geomean vs zero/negative.
   - Baseline **load + extraction** helpers (for the deferred dev/cloud baseline). **No
     `reference.json` committed** — when absent, scoring returns raw-times mode (headline omitted).
2. `reporting.py`:
   - **§7.3 plain-text report first** (canonical): pure ASCII, ≤72 cols, deterministic ordering,
     integer category/overall scores, 2dp per-task score, 3-sig-fig median, 2dp cv; engine suffixes
     `[pandas]/[polars]`; NOTES (noisy/excluded/p-e isolation); raw-times mode when no baseline.
     Conditional `--sweep`, hetero E-CORE block, `--summary`, non-standard threads.
   - Then `rich` console table, then `--format md|html`. Written next to JSON + echoed to console.
   - `compare`: refuse on mismatched `benchmark_version`/`reference_version`/`mode`; backend-
     sensitive vs neutral segmentation; checksum diff is informational, not a refusal.

**Test/checkpoint:** unit-test scoring math against a **small in-repo fixture baseline** (geomean,
category order, exclusions, zero-guard, bucketing, e_core_delta) — exercises the headline path
without committing a fake reference. Snapshot-test the txt report in raw-times mode for column
alignment / ≤72 col / determinism. `cpubench report FILE` re-renders; `cpubench compare A B` gates.

---

## Cross-cutting: `tests/` and tooling

- `tests/` — per-task quick-mode shape/dtype smoke (primary guard) + harness unit tests
  (environment dict, registry order, thread-env, affinity detection mocks, runner read-only,
  controller resume/timeout, scoring math vs fixture, report snapshot).
- `uv run pytest`, `uv run ruff check . && uv run ruff format .` green.
- Commit in **small reviewable steps**, one layer (and within Layer 5, one category) per commit,
  on a feature branch off `main` (only when the user asks to commit/push).

---

## Verification (end-to-end Definition of Done)

Run on this box (Linux/WSL2, 4 cores, 8 GB):

```bash
uv sync
uv run cpubench env                       # correct CPU/RAM/cores/BLAS, homogeneous
uv run cpubench list                      # tasks/categories/sizes from registry, fixed order
uv run cpubench run --mode quick          # green across all 6 categories
uv run cpubench run --mode quick --resume # second run skips completed partials
uv run pytest                             # shape/dtype smoke + harness/scoring/report units pass
uv run ruff check . && uv run ruff format .
```

Expected: resumable per-config partial files stream to `results/.partial/`, a valid **schema-v1**
results JSON is assembled, and the **§7.3 txt report renders in raw-times mode** (headline
omitted, `OVERALL (no baseline — raw times only)`, category lines omitted). `--mode normal` is
**not** a gate here (will OOM-fail large tasks on 8 GB — that's the spec's "measure then tune"
signal, recorded in `peak_rss_mb`, not a bug). P/E and macOS/Windows paths verified via mocked
unit tests only.

**Deferred, do NOT fake (SPEC §8/§13/§14):** `baselines/reference.json` (dev placeholder later,
cloud SKU before release — a `reference_version` bump); size tuning for FE panel / `md_gpr` /
`md_rf_predict` / `sp_fhash` (measure-then-tune from observed RSS); macOS Accelerate detection
confirmation on a real Apple box.

---

## Execution via `/goal` (autonomous build)

`/goal` is a built-in Claude Code command (v2.1.139+): a session-scoped autonomous loop that
keeps working turn-by-turn until a **transcript-verifiable** completion condition is met (a
Haiku evaluator checks after each turn, feeds back why it's not done, and re-runs). This plan's
Definition of Done is a perfect fit — three commands with visible exit codes.

**Chosen approach:** a *single end-to-end DoD goal* (not staged per-layer), run in
**auto-accept mode** for permissions (no settings allowlist needed).

### Beforehand
1. **Confirm Claude Code ≥ v2.1.139.** `/goal` is unavailable if `disableAllHooks` /
   `allowManagedHooksOnly` is set — its evaluator is a prompt-based Stop hook.
2. **Enable auto-accept mode** so `uv`/`git`/`pytest`/`ruff` calls don't prompt mid-loop.
3. **De-risk Layer 1:** run `uv lock && uv sync` once first so a dependency-resolution hiccup
   doesn't burn autonomous turns (§10 says the lock resolves cleanly post-numba/UMAP removal).
4. **Anchor is `PLAN.md` + `SPEC.md`** — the loop reads both; PLAN.md drives layer order,
   SPEC.md is authoritative for task details.

### The goal condition (paste verbatim)
```
/goal Implement ml-cpu-bench per PLAN.md (repo root), building in its Layer 1-6 order and
honouring the timing/threading/determinism invariants. DONE only when all three are shown in
this transcript with their output/exit codes: (1) `uv run cpubench run --mode quick` completes
covering all 6 categories (data_prep, linalg, factorization, clustering, models, sparse) and
writes a schema-v1 results JSON plus the SPEC §7.3 plain-text report in raw-times mode; (2)
`uv run pytest` passes (shape/dtype smoke + harness/scoring/report unit tests); (3)
`uv run ruff check .` reports no errors. Constraints: do NOT create or fake
baselines/reference.json; normal-mode OOM on this 8GB box is expected, not a failure. If the
same error blocks progress for 3 consecutive turns, stop and report it. Stop after 50 turns
regardless.
```

### Why this works / watch-outs
- The evaluator only reads the **transcript**, so the loop must actually *run* the three
  verification commands and show their output — the goal text forces that. Early turns read
  "not done, keep building" (cpubench doesn't exist yet), which is intended.
- The **turn cap (50)** and **3-strike same-error** clauses bound a runaway loop on a big build.
- The **"don't fake the baseline"** clause preserves the deferred-baseline invariant.
- Monitor with bare `/goal` (status: elapsed, turns, tokens, last reason); `/goal clear` to stop.

### Fallback if the single goal proves too long to steer
Staged per-layer goals (one `/goal` per layer, e.g. *"Layer 1 done when `uv run cpubench env`
prints CPU/RAM/cores/BLAS and its unit test passes"*), then a final DoD goal as the end-to-end
gate. A clean checkpoint-driven recovery, not the default path.
