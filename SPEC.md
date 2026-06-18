# ml-cpu-bench — Specification

A reproducible benchmark that measures **CPU** performance on classical machine-learning and
data-science workloads. No neural nets, no GPU. Clone, run one command, get per-task and
summarised results.

- **Status:** design spec (pre-scaffold)
- **Schema version:** `1`
- **Primary question it must answer well:** how good is a given CPU/system "as is" at
  classical ML + data-prep work, including a fair look at Apple Silicon.

---

## 1. Design principles

1. **Measure the system as shipped.** Detect and report the math backend (BLAS/LAPACK,
   OpenMP); never require the user to configure it. "As shipped" refers to the **software
   stack** — the backend and library versions the machine actually has — not the thread count.
   Thread count is a *controlled* variable (§2, §3), chosen to extract the machine's best
   honest whole-system throughput rather than observed from whatever each library happens to
   default to.
2. **Reproducible.** Fully synthetic data with fixed seeds; pinned dependencies via a
   universal `uv` lockfile; recorded environment metadata on every run; **deterministic task
   order** (tasks always run in registry order, never dict-iteration order) so a given machine
   produces the same sequence run-to-run.
3. **Fair across parallelism layers.** A single `--threads` setting is honoured by *all*
   parallel libraries (BLAS, scikit-learn, LightGBM, Polars, numexpr), not just NumPy.
4. **Isolated and honest timing.** Each task config runs in its own subprocess so thread
   settings take effect (they must be set pre-import), state never leaks between tasks, and
   peak memory is attributable to one task.
5. **Whole-machine is the headline; per-core is the diagnostic.** The **all-cores score is the
   primary number** — "how fast is the whole machine". The single-core score ("how good is each
   core") is a secondary diagnostic, computed only under `--sweep`. The report leads with
   all-cores and never blends the two.

---

## 2. Threading model

Two orthogonal axes: **how many** threads (`--threads`) and **which cores** (`--cores`).

### 2.1 Thread count

| Mode | Flag | What runs | Purpose |
|---|---|---|---|
| All cores | *(default)* | every task at `threads = detected_physical_cores` | whole-machine throughput |
| Single core | `--threads 1` | every task at 1 thread | per-core quality |
| Sweep | `--sweep` | each task at `1` and at `all` | scaling efficiency |
| Explicit | `--threads N` | every task at N threads | custom |

**Why physical, not logical, cores by default.** The goal is the machine's best *honest*
whole-system throughput, and physical cores deliver it without sabotaging any one library. For
compute-bound work, physical cores *is* full utilization in the only sense that matters —
throughput: dense BLAS at the physical-core count runs faster than at the logical count,
because the SMT/hyperthreading siblings add no extra FP/vector units, they only contend for the
one that's there. Pushing such tasks to logical cores would *reduce* throughput while reporting
it as "the whole machine" — exactly the deoptimization we refuse to do. (This is why OpenBLAS
and MKL themselves default to physical cores.) The honest cost is narrow: a few latency-bound
parallel tasks could squeeze single-digit-to-modest gains from SMT, but the tree models are
compute-bound and the data-prep tasks are memory-bandwidth-bound, so most of the suite is
already at or near peak at physical. Physical is therefore optimal-or-near-optimal across the
board *and* more reproducible (no SMT scheduling jitter). Both counts are detected and reported
(`psutil.cpu_count(logical=False)` vs `True`); `--threads N` reaches logical or any other count
for sweeps, which is also how you'd probe where SMT genuinely helps. On Apple Silicon there is
no SMT, so physical = all P+E cores; on Intel hybrid, physical = P-physical + E-physical
(E-cores have no SMT).

**Single-thread core placement.** A `--threads 1` run pins the single worker thread to **one
P-core** (hard affinity on Linux/Windows; high QoS `USER_INITIATED` on macOS). Without this,
the OS may schedule the lone thread on an E-core on heterogeneous chips, making the
"per-core quality" number wobble for reasons unrelated to the CPU. The single-core score
therefore means *one P-core*, and the report says so.

### 2.2 Core selection (`--cores all|p|e`, default `all`)
Only meaningful on heterogeneous CPUs (Apple Silicon; Intel hybrid, 12th-gen+). On
homogeneous CPUs `p`/`e` are no-ops equal to `all`, and the report omits the p-core / E-core
sections entirely rather than emitting null rows.

**Detection fallback — when in doubt, homogeneous.** P/E detection is best-effort. If the
platform exposes no usable signal (no sysfs `cpu_capacity` / core-type flags on Linux, no
CPU-set efficiency class on Windows, no `hw.perflevel*` on macOS) **or** the signal reports a
single uniform core type, the CPU is treated as **homogeneous**: `perf_cores` / `eff_cores`
are recorded as `null`, `--cores p`/`e` collapse to `all`, and the heterogeneous-only blocks
(`p_cores`, `e_core_delta`, the E-CORE CONTRIBUTION section) are omitted. We never *guess* a
P/E split from core counts or model strings — an uncertain detection is reported as
homogeneous rather than producing an unreliable split.

**Enforcement differs by OS, and the runner reports what it actually achieved, never what it
requested.** `affinity.py` exposes enforcement as a **modular per-platform backend**
(detect → apply pin/QoS → verify residency), so the soft macOS P-core path (see §13) can be
improved later without touching the runner, scoring, or report:

- **Linux / Windows — hard pinning (enforced).** Detect P vs E cores (Linux: sysfs
  `cpu_capacity` / core-type flags; Windows: CPU-set efficiency class) and pin via
  `sched_setaffinity` / `SetThreadAffinityMask` (both wrapped by `psutil.cpu_affinity`).
  `--cores p` means strictly P-cores.
- **macOS — best-effort, no hard affinity.** macOS exposes no API to pin a thread to a core
  or core *type*; `psutil.cpu_affinity` is unavailable there. The only lever is QoS class, and
  because each config already runs as its own subprocess we set it **at spawn time via the
  `taskpolicy(8)` CLI** rather than reaching into `libSystem` from Python (no `ctypes`, no
  sudo):
  - `--cores e` ⇒ launch the worker under `taskpolicy -b` (background QoS). macOS confines
    background work to E-cores fairly reliably, so E-cores are actually *easier* to isolate
    than P-cores on Apple Silicon.
  - `--cores p` ⇒ run at the default/`USER_INITIATED` QoS (no `taskpolicy` flag) with the
    thread count capped at the P-core count. There is **no shell lever that forces P-cores**;
    default QoS merely *biases* toward them in steady state. This is reported as `biased`, not
    a guarantee, and is checked by residency sampling below.
  - Single-thread P placement (§2.1) uses the same default-QoS path; on macOS it is best-effort.
- **Verification (all platforms, macOS especially).** During each timed region the runner
  measures the fraction of work that landed on the "wrong" core type via `psutil.cpu_times(percpu=True)`
  (sysfs/PDH/`host_processor_info` underneath — no sudo). Concretely: a low-priority sampler
  thread takes per-core busy-time deltas at a fixed **10 Hz** (100 ms) cadence, **active only
  while `ctx.timer()` is open**. On Linux/Windows the sampler thread is pinned to the
  **complement** of the cores under test so it does not steal a test core; on macOS it runs
  unpinned at background QoS. `offcore_residency_pct = busy_on_wrong_type / total_busy` over
  the timed window. Each result carries an `enforcement` field: `none` (cores=all, no isolation
  requested), `pinned` (hard affinity), or `biased` (QoS), plus the measured
  `offcore_residency_pct`. A run with `enforcement="biased"` and `offcore_residency_pct > 10%`
  is flagged **leaky** in the report so the number can be discounted rather than trusted blindly.

### 2.3 Pre-import thread control
Thread *count* is enforced in the worker subprocess **before any heavy import**, by setting:

- `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`,
  `VECLIB_MAXIMUM_THREADS` (Apple Accelerate), `NUMEXPR_MAX_THREADS`
- `POLARS_MAX_THREADS` — **must be set to the chosen thread count.** Polars otherwise defaults
  to *logical* cores, which would silently give it more threads than the rest of the suite and
  make the pandas-vs-Polars comparison unfair even at the all-cores default.
- and at call time: scikit-learn `n_jobs`, LightGBM `num_threads`, plus a
  `threadpoolctl.threadpool_limits` context as a belt-and-braces guard around BLAS regions.

> **Apple Silicon note.** Physical-core count includes efficiency (E) cores. The default
> all-cores run mixes fast P-cores and slow E-cores. P/E split is detected via
> `sysctl hw.perflevel0.physicalcpu` / `hw.perflevel1.physicalcpu`. Because true P-isolation
> isn't enforceable on macOS, evaluate Apple Silicon from the **triplet** below rather than a
> single P-only number.

### 2.4 E-core contribution (the headline Apple-Silicon question)
For heterogeneous chips the report computes, per task, whether the E-cores help or hurt total
throughput, by comparing `all-cores` time against the best-effort `p`-cores time:

```
e_core_delta = (p_cores_median_s - all_cores_median_s) / p_cores_median_s
```

- `e_core_delta > 0` (all-cores meaningfully faster than `p`) ⇒ E-cores add throughput
  (load-balanced workload).
- `e_core_delta ≈ 0` or `< 0` (all-cores ≈ `p` or slower) ⇒ E-cores are stragglers at sync
  barriers (no gain / net harm).

This per-task `e_core_delta` is a **direct intra-machine ratio and needs no reference**. It is
reported alongside the single-core score and the direct E-core throughput measurement (from
`--cores e`), giving the triplet — single-P-core / E-core / all-cores — that actually
characterises the chip.

> **Producing the all-cores/p-cores pair in one run.** On a detected heterogeneous CPU, a
> single `cpubench run` auto-expands to the configs needed for the cross-machine block
> (all-cores and p-cores; plus e-cores when requested), so the user does not have to stitch
> multiple invocations together. The **single-P-core** leg of the triplet is still gated
> behind `--sweep` (consistent with §1.5/§2.1/§8): a plain run reports the E / all-cores
> pair and the p-core overall, and the full single-P / E / all-cores triplet appears only
> when `--sweep` is also passed.

---

## 3. Architecture

```
controller (cli)
  └─ for each (task, size, threads, cores) config, in deterministic registry order:
       └─ spawn worker subprocess
            • set thread env vars  (pre-import)
            • import libs, generate data (untimed)
            • warm-up run (discarded; default ON)
            • fixed repetitions (timed)
            • write one JSON result FILE (not stdout)
  └─ collect result files → compute scores → write report
```

**Subprocess isolation** is the key choice: it is the only robust way to honour the
pre-import thread env vars, and it gives clean per-task RSS and no cross-task cache/state
contamination.

### Timing rules
- Wall-clock via `time.perf_counter()`. CPU time via `time.process_time()` recorded
  secondarily (reveals parallelism: CPU≫wall ⇒ multi-threaded).
- **Data generation, imports, and allocation are outside the timed region.** Only the
  algorithm under test is timed.
- **Warm-up:** 1 discarded run per config. **Default ON** — it absorbs first-call costs
  (cache warming, lazy allocation) so they don't contaminate rep 1.
- **Fixed repetitions:** run exactly `repeat` timed reps (default 5). No adaptive logic and no
  per-task time budget; tasks run to completion however long they take.
- **No reset between reps.** Warm machine and library state (CPU caches, the memory allocator,
  BLAS thread pools) is deliberately *not* reset between the timed reps of a config — steady
  state is what we measure, and the warm-up run already absorbed the cold start. The only
  requirement is that every rep does **identical work**, which means the timed region treats its
  input as **read-only**: non-mutating ops (`np.sort`, not `arr.sort`; `df.sort_values`, not
  in-place), no destructive flags (scipy `overwrite_*` off, `copy_X=True`), and
  `warm_start=False` so each `.fit()` fully retrains rather than continuing. Reps are therefore
  naturally independent and must not accumulate memory across iterations. Where an operation is
  inherently destructive (an unavoidable in-place op, or a solver that overwrites its matrix),
  only the **input** is restored, **outside** the timed region, between reps — a minimal data
  restore, never a warm-state reset. (Concrete trap: `dp_sort` must use a non-mutating sort, or
  reps 2+ would time an already-sorted input, which adaptive sorts finish far faster.)
- **Reported per config:** median, min, std, coefficient of variation (CV), all raw reps,
  peak RSS (MB), CPU-time/wall ratio. CV above 0.10 sets a `noisy` flag.
- **Inter-task buffer** (`--cooldown SECONDS`, default 2) between tasks: a settling gap for
  subprocess teardown, memory release, and OS quiescence between configs. Note this is *not*
  a thermal-control mechanism — throttling that occurs during a task is intentionally retained
  as real-world system behaviour and reflected in the result.

### Worker → controller protocol
- The worker writes its result as JSON to a **dedicated file** (`results/.partial/<config_id>.json`),
  never to stdout. stdout/stderr are reserved for library chatter (LightGBM, OpenMP banners,
  BLAS info, warnings), which would otherwise corrupt a stdout-parsed result line. The
  presence of the partial file is also what `--resume` checks to skip completed configs.
- **Per-config timeout (`--timeout SECONDS`, default 3600 = 1 h).** A generous wall-clock
  ceiling whose *only* purpose is to stop a hung or pathologically slow worker from blocking the
  whole suite — **not** a per-task time budget (reps still run to completion). On timeout the
  controller kills the worker and records `status: "failed"` (no result file), then moves on;
  `--resume` re-attempts nothing already marked done. `--timeout 0` disables it. Sizes are still
  tuned so every config finishes comfortably inside this ceiling on the range of target
  machines; the timeout is the safety net, not the schedule. The same `failed` path covers
  OOM-kill (SIGKILL) and any non-zero worker exit.

### Memory safety
- **Headroom by design.** Large-tier sizes target peak RSS ≤ ~8 GB. Target machines typically
  have **> 16 GB RAM**, so this leaves ample headroom.
- **No pre-task estimate guard.** There is no `mem_estimate` and no `skipped_memory` status.
  Instead, every config records its observed `peak_rss_mb`, and we **tune sizes down to the
  target machines from those measurements** ("measure then tune"). A task that is genuinely too
  big for a machine simply OOM-fails (`status: "failed"`) and is excluded; it does not hang the
  suite.
- **Swap detection.** The runner samples swap-in/out counters around the timed region; if the
  OS swapped during a task, the result is flagged `swapped: true` and excluded from scoring,
  since it measures the disk, not the CPU.

### Other harness behaviours
- **Targeted imports.** Each worker imports only the libraries its task needs (e.g. the
  LightGBM stack is never imported for a linalg task), so slow imports don't tax the budget.
- **Per-task checksum (optional, informational).** A task *may* return a cheap summary value
  (e.g. KMeans inertia, model loss, output-array shape + a summary statistic), stored per
  result. This is **informational only** — we are after high-level speed measures, not
  numerical validation, so there is **no per-task tolerance calibration** and `compare` does
  **not refuse** on a checksum difference (it surfaces any difference as a note; see §6).
  Correctness is guarded instead by the `tests/` smoke suite, which asserts each task returns
  the expected output shape/dtype in `quick` mode (§11).
- **Incremental, resumable output.** The **per-config partial files** (`results/.partial/<config_id>.json`,
  one per completed config) *are* the incremental stream and the single source of truth for
  resume — there is no separate JSONL log. As each worker finishes it leaves its partial file;
  `--resume` decides what to skip purely by which partial files already exist (a `failed`
  config leaves no file and is re-attempted). At the end of the run the controller assembles
  the authoritative schema-v1 JSON (§7.1) from all partial files. A crash or OOM partway
  through a long run therefore preserves everything already done.
- **Quiet-machine pre-flight.** Before starting, sample system-wide CPU load; warn if the
  machine isn't idle and record the baseline load in metadata.

---

## 4. Run modes

A single `--mode` selects both which tasks run and how big. Two modes:

| Mode | Tasks | Sizes | Typical use |
|---|---|---|---|
| `quick` | fast subset (excludes the long-runners tagged "normal only" in §5: `md_gpr`, `md_lgbm_multi`, `nlp_lda`, `sp_lasso_cv`) | `quick` column | CI / sanity |
| `normal` *(default)* | every task | `normal` column | the real benchmark |

Both modes now cover the **same 6 categories** (data-prep, linalg, factorization, clustering,
models, sparse), so quick and normal headlines are structurally comparable. `normal` sizes are
chosen so peak RSS stays **≤ ~8 GB**, comfortable on 16 GB and roomy on the typical > 16 GB
target machine. The runtime memory guard (§3) skips any task that would exceed available RAM on
smaller machines, and `--tasks` / `--exclude` still select individual tasks within a mode.

> Wall-clock duration is left unspecified: it depends on the machine and on the chosen tasks,
> sizes, and `repeat` count. These are measured on real runs and tuned afterwards rather than
> predicted here. Note only that subprocess isolation makes every config re-pay its library
> import cost; that cost is recorded per worker (`import_time_s`) so it stays visible.

---

## 5. Task catalogue

All data is synthetic, fixed-seed (`--seed`, default `1337`). scikit-learn / LightGBM
estimators are also seeded with `random_state = seed` (1337) so the *work* — tree structure,
init, solver path — is stable run-to-run. Each size cell is written `quick → normal`. All
arrays use **explicitly pinned dtypes** (float64 unless noted; integers always `int64`, never
the platform-default `long`) so the generated data is identical across OSes and architectures
(see §11). Each task records the exact parameters used.

### 5.1 Data preparation & feature engineering — run on **both pandas and Polars** (identical logical op)
Two groups of tasks, all in the `data_prep` category (kept as one category — feature engineering
is folded in here rather than given its own, so a single workload family can't carry outsized
weight in the headline geomean). The report shows pandas vs Polars side by side for every task.

**(A) Core data prep.** Source frame generated **in memory** (untimed) — mixed dtypes: 20
numeric, 3 categorical (low/med/high cardinality), 1 datetime. No disk I/O is timed; read/parse
performance is out of scope.

| Task | Operation | Key sizes — quick → normal (rows unless noted) |
|---|---|---|
| `dp_groupby` | group by categorical, agg sum/mean/std/count on 4 cols | 2M (1k groups) → 10M (50k groups) |
| `dp_join` | inner join two frames on key | 0.5M⋈0.5M → 8M⋈2M |
| `dp_sort` | multi-key sort (2 columns) | 2M → 10M |
| `dp_filter` | boolean mask + column select | 2M → 10M |
| `dp_string` | regex extract + lower + contains on string col | 1M → 6M |
| `dp_rolling` | **global** rolling (whole frame, sorted by datetime): mean, std, median, window=100 | 1M → 6M |

**(B) Feature engineering — leakage-safe time-series panel features.** A synthetic panel frame
is generated and sorted by `(entity_id, timestamp)` **untimed** before the timed region. Schema:
`entity_id` (int64), `timestamp` (datetime64[ns]), `target` (float64), 20 numeric features
(float64), 3 categorical features (low/med/high cardinality). All FE tasks share the same panel
size per mode: **quick 2M rows / 2k entities → normal 10M rows / 20k entities** (bounded well
below the proposal's 50M to stay within the RSS target — grouped rolling in pandas is the memory
and time driver). Every window/expanding/encoding computation uses **only prior observations**
(`closed="left"` semantics): no row may use its own value or any future row.

| Task | Operation (per entity unless noted) | Distinct pattern it stresses |
|---|---|---|
| `fe_lags` | grouped lags: `lag_1`, `lag_7`, `lag_30` (grouped `shift`) | grouped gather / shift |
| `fe_rolling` | grouped **leakage-safe** rolling: mean (7/30/90) + std (30), `closed="left"` | grouped window functions (heaviest) |
| `fe_expanding` | grouped expanding mean + historical group normalization (`value ÷ mean of prior obs`) | grouped cumulative / expanding |
| `fe_ewm` | grouped exponentially weighted moving average | recursive (non-windowed) accumulation |
| `fe_onehot` | one-hot / dummy encoding of the low- and medium-cardinality categoricals (`pd.get_dummies` / Polars `to_dummies`), output `uint8` | dense column expansion (memory-bound) |
| `fe_rank` | **cross-sectional** rank + z-score per `timestamp` (group by time across entities) | many-small-groups rank (different group axis) |
| `fe_datetime` | calendar features (day_of_week, month, quarter) + cyclical sin/cos + vectorized `log1p`/`clip`/`where` | cheap vectorized arithmetic |

> Join keys are generated with **controlled cardinality (~1:1)** so the result set stays close
> to the input size — never a many-to-many blow-up that would explode memory rather than test
> join throughput.

> Polars is multi-threaded by default and pandas largely is not. At all-cores this is a fair
> real-world comparison; at `--threads 1` Polars is pinned via `POLARS_MAX_THREADS` so the
> single-core comparison is apples-to-apples. The report presents pandas vs Polars per task
> explicitly. The source frame/panel is generated **once** (untimed) and both engines build their
> native frame from the same arrays, so they operate on identical input; frame construction is
> outside the timed region.

> **FE fairness & correctness.** (1) The pandas and Polars variants should compute the *same
> logical feature* (same windows, lags, `closed="left"` intent, `min_periods`), but **exact
> numerical equivalence between the engines is not required or asserted** — start-of-group NaN
> handling and last-bit differences are accepted, since the two are scored as separate entries
> and we measure speed, not agreement. (2) The FE pipeline adds many columns, which is an
> in-place-mutation trap under the no-reset rule (§3): each rep must build features into a fresh
> structure, never mutate the shared panel. (3) Any checksum returned is informational (§3) —
> the two engines may differ, which is fine.

### 5.2 Dense linear algebra — NumPy / SciPy (BLAS/LAPACK stress)

| Task | Op | Sizes — quick → normal |
|---|---|---|
| `la_gemm` | `A @ B`, dense square | N = 2000 → 8000 |
| `la_solve` | `scipy.linalg.solve` (LU) | N = 2000 → 6000 |
| `la_cholesky` | `cholesky` on SPD | N = 2000 → 6000 |
| `la_qr` | `qr`, tall matrix | 3000×1500 → 10000×4000 |
| `la_svd` | `svd(full_matrices=False)` | 2000×1000 → 6000×3000 |
| `la_eigh` | symmetric eigendecomposition | N = 1500 → 4500 |
| `la_fft` | `scipy.fft.fft`, 1D complex | 2²² → 2²⁵ |

### 5.3 Matrix factorization — scikit-learn

| Task | Op | Sizes (n_samples × n_features), components — quick → normal |
|---|---|---|
| `mf_tsvd` | `TruncatedSVD` | 20k×500 → 100k×2000, k=50 |
| `mf_pca` | `PCA` (full) | same shapes, k=50 |
| `mf_nmf` | `NMF` (nndsvda, max_iter 200) | 5k×500 → 20k×2000, k=20 |

### 5.4 Clustering — scikit-learn

| Task | Op | Sizes — quick → normal |
|---|---|---|
| `cl_kmeans` | `KMeans` (n_init=3, max_iter=100) | 100k×20, k=25 → 600k×50, k=100 |
| `cl_mbkmeans` | `MiniBatchKMeans` (batch 1024) | 300k, k=25 → 3M, k=100 |
| `cl_optics` | `OPTICS` (min_samples=10, ball_tree, bounded `max_eps`, `n_jobs` honoured) | 10k → 30k × 10 *(O(n²)-bounded)* |
| `cl_gmm` | `GaussianMixture` (full cov) | 50k×10, comp 10 → 200k×10, comp 30 |

> `cl_optics` replaces the earlier DBSCAN slot: same density-based family, but it computes the
> full reachability ordering, so it is pricier per point than DBSCAN. `normal` is bounded a bit
> lower (30k vs the 40k DBSCAN used) to keep the O(n²) neighbour work in check; a finite
> `max_eps` with the ball-tree index keeps it from degenerating to the full O(n²) distance
> matrix. Checksum is a tolerant invariant (cluster count + ordering hash).

### 5.5 Model fitting — scikit-learn + LightGBM + statsforecast

| Task | Op | Sizes (n × features) — quick → normal |
|---|---|---|
| `md_linreg` | OLS `LinearRegression` | 200k×100 → 1M×300 |
| `md_ridge` | `RidgeCV` | same as linreg |
| `md_lasso` | `LassoCV` (coordinate descent) | same as linreg |
| `md_logreg` | `LogisticRegression` (lbfgs, 3-class) | 200k×50 → 800k×150 |
| `md_bayes_ridge` | `BayesianRidge` (Bayesian linear regression, ARD-style hyperprior) | 200k×100 → 1M×300 |
| `md_gpr` | `GaussianProcessRegressor` (RBF kernel, `optimizer=None`, 1000 predict pts) *(O(n³) kernel Cholesky — backend-sensitive; `normal` only)* | n_train 2k×20 → 10k×20 |
| `md_rf` | `RandomForestClassifier` | 100k×50, 100 trees → 200k×50, 300 trees |
| `md_rf_predict` | `RandomForestClassifier.predict` — forest **trained untimed** on 200k×50 / 300 trees; timed region scores a large `X_test` (`n_jobs` honoured over trees) | predict rows 500k×50 → 5M×50 |
| `md_hist_gbm` | `HistGradientBoostingClassifier` | 200k×50, 100 iters → 1M×100, 300 iters |
| `md_lgbm` | LightGBM, **regression** (`num_threads` honoured) | 200k×50, 100 trees → 500k×100, 500 trees |
| `md_lgbm_multi` | LightGBM, **multiclass softmax, `num_class=10`** *(builds num_class × rounds trees; `normal` only)* | 200k×50, 250 rounds *(normal only)* |
| `md_svc_rbf` | `SVC` RBF *(single-threaded dominator — kept small)* | 5k×30 → 15k×30 |
| `md_knn` | `KNeighborsClassifier` brute, 1000 queries | 50k×50 → 200k×50 |
| `md_autoarima` | statsforecast `AutoARIMA` (nonseasonal, `season_length=1`) fit across a panel of independent series with 1 exogenous regressor (relevant for ~half, noise for the rest), forecasting `horizon` steps; parallel over series via `n_jobs`. *(numba-JIT — warm-up absorbs first-call compile)* | 8 series × 96 pts, h=1 → 1024 series × 1024 pts, h=24 |

> `md_rf_predict` measures **inference**, a distinct workload from training: tree
> traversal that is cache-, latency-, and branch-prediction-bound rather than compute-bound, and
> the common production case where a model is trained once and scores many rows. The forest is
> trained untimed on a modest set (the same 200k×50 / 300-tree shape as `md_rf`) so
> the untimed setup isn't pathological; only `predict` on the large `X_test` is timed, and it
> parallelizes over trees via `n_jobs`. Checksum is `{rows, prediction_sum}` — exact and
> deterministic on a given build (cross-architecture it may differ if the trained forest differs
> by an FP-tie split, which is the accepted tolerance, not a bug).

> `md_gpr` is the heavy Bayesian method in the suite. With `optimizer=None` (fixed RBF
> hyperparameters) the fit is a single dense Cholesky of the `n×n` kernel matrix plus a solve —
> O(n³) compute, O(n²) memory — so it doubles as a real BLAS stress applied through scikit-learn
> rather than NumPy. The fixed kernel keeps it deterministic and reproducible; `n_train` is held
> to 10k at `normal` so the kernel (10k² × 8 ≈ 0.8 GB, plus working copies) stays well under the
> RSS target. It is excluded from `quick` (O(n³)). **Because its time is dominated by the kernel
> Cholesky, `md_gpr` is tagged backend-sensitive for `compare` even though it lives in the
> models category** (see §6). Checksum is the rounded log-marginal-likelihood.

> `md_lgbm_multi` data is 10 Gaussian class blobs (centroids drawn via
> `default_rng`, then sampled around them) so the classes are learnable and stable across
> library versions. At `num_class=10` × 250 rounds it builds ~2,500 trees (more than the
> binary/regression task); it is excluded from `quick`. Its
> correctness checksum is a tolerant invariant (final multiclass log-loss rounded), since
> LightGBM is non-deterministic across thread counts.

### 5.6 Sparse & NLP workloads — SciPy sparse + scikit-learn
Exercises the sparse code paths that classical ML and text work actually hit — memory-bound,
irregular access, and largely independent of the dense BLAS backend (so these land in the
backend-neutral group for `compare`). Sparse matrices are synthesized as CSR with controlled
density via `scipy.sparse.random(..., random_state=rng)`, or a synthetic Zipfian token corpus
for the text tasks — all seeded for stability. No new dependencies.

| Task | Op | Sizes (n_samples × n_features), density — quick → normal |
|---|---|---|
| `sp_tfidf` | `TfidfVectorizer.fit_transform` on a synthetic Zipfian token corpus | 50k docs → 300k docs; vocab 30k; ~80 tokens/doc |
| `sp_hashvec` | `HashingVectorizer.transform` (n_features=2²⁰) on the Zipfian token corpus | 50k docs → 300k docs; ~80 tokens/doc |
| `sp_fhash` | `FeatureHasher` (input_type='string', n_features=2²⁰) on high-cardinality categorical rows | 200k×20 fields → 1M×30 fields; cardinality ~1e6/field |
| `sp_matmul` | sparse CSR × dense product (SpMM primitive) | 100k×30k → 200k×50k @ 0.2%, dense rhs × 64 |
| `sp_tsvd` | `TruncatedSVD` (randomized) on sparse CSR (LSA use case) | 50k×10k → 200k×50k @ 0.3%, k=100 |
| `sp_nmf` | `NMF` (k=20) on a sparse TF-IDF doc-term matrix (topic model) | 20k×10k → 100k×30k @ derived density |
| `sp_saga` | `LogisticRegression` (saga solver) on sparse input, binary | 50k×10k → 200k×50k @ 0.3% |
| `nlp_lda` | `LatentDirichletAllocation` (n_components=20, batch, `n_jobs` honoured) on a sparse doc-term matrix *(variational Bayes; `normal` only)* | 20k docs → 100k docs; vocab 10k → 30k |
| `sp_lasso_cv` | `LassoCV` (cv=5, n_alphas=50, `n_jobs` honoured) on sparse regression data *(CV multiplies work; `normal` only)* | 20k×1k → 100k×10k @ 0.5% |

> `sp_nmf` runs NMF on a sparse TF-IDF matrix (the LSA/topic-model use case), which is a
> genuinely different code path from the dense `mf_nmf` — sparse multiplicative/coordinate
> updates rather than dense BLAS — so the two don't duplicate each other.

> `sp_hashvec` and `sp_fhash` both exercise the hashing-trick path: a
> stateless `transform` (no `fit`, no vocabulary), deterministic (MurmurHash), and
> **single-threaded** in scikit-learn — so, like `md_svc_rbf`, they measure raw per-core
> hashing throughput even inside an all-cores run. The vectorizer hashes a token stream; the
> feature hasher hashes `"field=value"` strings, the realistic way high-cardinality categorical
> data is encoded into a fixed sparse space without building a category dictionary. Because the
> transform is deterministic, their checksums are **exact** (output shape + nnz), not tolerant.
>
> **`sp_fhash` sizing caveat.** Its input is an iterable of per-row feature lists;
> keeping data generation untimed means materializing that input, which for high-cardinality
> data is millions of short Python strings — and *that* materialization, not the compact sparse
> output, is the real memory driver. Sizes are bounded with that in mind (e.g. `normal` is
> 1M×30 ≈ 30M tokens), and its `peak_rss_mb` will be dominated by that materialized input rather
> than the compact sparse output — so size-tuning watches the input, not the output matrix.

> `nlp_lda` is topic modeling by variational Bayes, so it also serves as a second Bayesian
> method (alongside `md_bayes_ridge` and `md_gpr`). The batch variant parallelizes the
> E-step across documents (`n_jobs`), giving another work-stealing probe for E-core
> contribution. Excluded from `quick`. Checksum is a tolerant invariant (rounded perplexity).

> `sp_lasso_cv` parallelizes across CV folds — a clean work-stealing workload, so it doubles
> as a strong probe for whether Apple's E-cores add throughput (the regime where they should
> *help*, unlike the barrier-bound dense factorizations). Given the CV cost, it's excluded from
> `quick`. Correctness checksums use tolerant invariants (selected alpha
> / final objective), as solver paths vary slightly across threads.

---

## 6. CLI

```
cpubench run            [--mode quick|normal]
                        [--threads N | --sweep] [--cores all|p|e]
                        [--tasks t1,t2,...] [--exclude t1,...]
                        [--repeat N] [--seed S] [--cooldown SECONDS]
                        [--timeout SECONDS] [--no-warmup] [--resume]
                        [--out results/<timestamp>.json]
                        [--format txt|md|html] [--summary] [--no-report]
cpubench list           # list tasks + categories + sizes
cpubench report FILE    # (re)render from a results JSON; [--format txt|md|html] [--summary]
cpubench compare A B    # diff two runs; see below
cpubench env            # print detected environment + backend only
```

Entry point installed as `cpubench` (also `python -m cpubench`). `--repeat` sets the fixed
number of timed reps (default 5); `--timeout SECONDS` is the per-config hang ceiling (default
3600; `0` disables); `--no-warmup` disables the default-on warm-up run. A
**plain-text report is written next to the JSON and printed to the console by default** (§7.3);
`--format md|html` additionally emits those (the flag is named `--format` on both `run` and
`report` for consistency); `--summary` restricts output to the shareable header + score block
(no per-task table); `--no-report` suppresses report generation entirely.

`compare` is **noise-aware and safe**: it refuses (or warns) when `benchmark_version`,
`reference_version`, or `mode` differ; flags a per-task regression only when the gap exceeds
the combined run-to-run variance of both runs; and segments the diff into **backend-sensitive**
vs **backend-neutral** groups so an Accelerate-vs-OpenBLAS comparison isn't misread as a CPU
difference. If both runs carry a per-task checksum and they differ, `compare` notes it
informationally (possible different work) but does **not** refuse — checksums are informational
(§3). Backend-sensitivity is a **per-task tag** (set in the registry), not inferred from
category: it covers linalg and dense factorization, plus `md_gpr` (whose time is dominated by
the dense kernel Cholesky). Tree models, data-prep, clustering, and the sparse/NLP tasks —
whose heavy work is the sparse matmul, not dense BLAS — are backend-neutral.

---

## 7. Output

### 7.1 Machine-readable JSON (authoritative)
```jsonc
{
  "schema_version": 1,
  "benchmark_version": "1.1.0",            // task/size definitions; compare refuses on mismatch
  "reference_version": "ref-2026.06",      // identity of baselines/reference.json; compare refuses on mismatch
  "run_id": "2026-06-14T03-22-10Z_ab12cd",
  "config": { "mode": "normal", "threads_mode": "all",
              // threads_mode ∈ { "all" (physical-all, the default), "single" (--threads 1),
              //   "explicit" (--threads N, neither 1 nor physical-all → no scored bucket) }.
              // --sweep runs both "all" and "single" as two separate configs; each result
              // carries its own threads_mode + threads. The heterogeneous auto-expanded
              // p-core pass is threads_mode "all" with cores="p" (it is NOT its own mode).
              "threads": 12, "cores": "all", "seed": 1337, "repeat": 5,
              "warmup": true, "cooldown_s": 2, "timeout_s": 3600 },
  "environment": {
    "cpu_model": "...", "arch": "arm64|x86_64",
    "logical_cores": 12, "physical_cores": 12,   // equal on Apple Silicon (no SMT)
    "perf_cores": 8, "eff_cores": 4,             // null when unknown (non-heterogeneous)
    "ram_gb": 32, "os": "Darwin 24.x", "python": "3.12.x",
    "blas_backend": "Accelerate|OpenBLAS|MKL", "blas_threads_detected": 12,
    "baseline_load_pct": 3.1,                    // system load sampled at pre-flight
    "import_time_s": 2.4,                        // representative worker import cost (informational)
    "libs": { "numpy": "...", "scipy": "...", "scikit-learn": "...",
              "pandas": "...", "polars": "...", "lightgbm": "..." },
    "continuous_run": true                       // tasks ran back-to-back (informational)
  },
  "results": [
    { "task": "la_gemm", "category": "linalg", "mode": "normal", "threads": 12,
      "cores": "all", "enforcement": "none",     // "none" (cores=all) | "pinned" (Linux/Win p|e) | "biased" (macOS p|e QoS)
      "offcore_residency_pct": 0.0,              // work that landed on the wrong core type (p|e runs)
      "params": { "N": 8000 }, "checksum": "shape=8000x8000;trace=1.2e4",
      "reps_s": [1.91, 1.88, 1.90, 1.89, 1.90], "median_s": 1.90, "min_s": 1.88,
      "std_s": 0.012, "cv": 0.006, "peak_rss_mb": 1620.4,
      "cpu_wall_ratio": 11.4, "swapped": false,
      "status": "ok", "noisy": false, "error": null }
      // status also: "failed"   (OOM / non-zero exit / manual interrupt)
    // ...
  ],
  "scores": {
    // headline = 100 × geomean(by_category values) — category-weighted, not task-weighted.
    // Only the canonical thread configs get a scored bucket:
    //   threads_mode "all" → all_cores ;  --threads 1 → single_core ;  --sweep → both.
    // A non-standard --threads N (neither 1 nor physical-all) has NO reference denominator,
    // so it produces raw times + intra-run ratios only and the "scores" block is omitted
    // (report labels it "threads=N (non-standard — raw times only)"). See §8.
    "all_cores":   { "headline": 142.0,
                     "by_category": { "linalg": 1.71, "clustering": 1.20, /* ...6 total */ },
                     "per_task": { "la_gemm": 1.55, /* ... */ } },
    "single_core": { "headline": 96.0,  "by_category": { /* ... */ }, "per_task": { /* ... */ } },  // if --sweep
    "p_cores":     { "headline": 138.0, "by_category": { /* ... */ }, "per_task": { /* ... */ } },  // heterogeneous; shares the all-cores reference denominator
    "e_core_delta": { "la_gemm": -0.02, "md_rf": 0.31 }     // >0 ⇒ E-cores help; intra-run, no reference
  }
}
```

### 7.2 Human summary
- Console table via `rich`: per-task median/CV/RSS, grouped by category, noisy rows marked,
  pandas-vs-Polars shown side by side.
- Headline block: the **all-cores score is the headline number**, presented first and largest.
  If swept, the **single-core score** and **scaling efficiency** follow as secondary
  diagnostics. The block notes that the all-cores headline scales with core count (it's a
  whole-machine number) and that single-core is the per-core property if you need to separate
  the two.
- **Category-score block — first-class, always shown immediately beneath the headline.** The
  six category scores are displayed as a labelled list (e.g. `Data prep & FE 155`, `Linalg
  171`, `Clustering 128`, …), not buried in the per-task table, since the headline is their
  category-weighted geomean and they're the most actionable summary. Example:
  ```text
  Overall (all cores)   142
  Category scores
    Data prep & FE      155
    Linalg              171
    Factorization       163
    Clustering          128
    Models              135
    Sparse / NLP        118
  ```
- Per-task table follows, grouped by category, with pandas-vs-Polars side by side.
- On heterogeneous chips: an **E-core contribution** table (per-task `e_core_delta`) and, for
  any best-effort (`biased`) runs, the measured off-core residency so leaky isolation is
  visible rather than silent.
- The console output uses `rich` (alignment, colour, optional bars). The canonical
  copy-pasteable artifact is the plain-text report defined in §7.3; `--format md|html` are
  optional extras.

### 7.3 Plain-text report (`--format txt`, default)
A deterministic, ASCII-only report designed to be pasted into a forum, issue, or gist for
side-by-side comparison. It is written next to the JSON and echoed to the console on every run.

**Format rules (so two reports diff cleanly):**
- **Pure ASCII**, monospace, **total width ≤ 72 columns**, no line ever wraps; rules are `=`
  (major) and `-` (minor).
- **Deterministic** content and ordering: tasks in registry order, categories in a fixed order,
  fixed rounding — so two pasted reports line up and `diff` is meaningful.
- **Number formats:** OVERALL and category scores are **integers**; per-task `score` (ratio
  vs reference) is **2 dp**; `median(s)` is **3 significant figures**; `cv` is **2 dp**. All
  numeric columns are right-aligned at fixed positions.
- **Engine variants** are suffixed `[pandas]` / `[polars]` and **each is scored separately**
  (both contribute to the `data_prep` category geomean) — there is no single blended data-prep
  number, by design.
- **Comparability** is stated in the footer: a comparison is only valid across runs with the
  same `benchmark_version`, `mode`, and (for the linalg/factorization rows) BLAS backend. The
  header surfaces all three so a reader can check at a glance.

**Layout — three blocks (identity → score summary → per-task), then notes:**

```text
========================================================================
 ml-cpu-bench 1.1.0      OVERALL  142          all-cores / normal
========================================================================
 CPU      Apple M3 Pro (arm64)
 Cores    12 physical / 12 logical   (8P + 4E)
 Threads  12   cores=all   enforcement=biased
 BLAS     Accelerate
 System   Darwin 24.3   Python 3.12.4
 Run      seed 1337   repeat 5   warmup on   ref ref-2026.06
------------------------------------------------------------------------
 SCORE SUMMARY                                       reference = 100
------------------------------------------------------------------------
   OVERALL  (all cores)                  142
   ....................................
   Data prep & FE                        155
   Linalg                                171
   Factorization                         163
   Clustering                            128
   Models                                135
   Sparse / NLP                          118
------------------------------------------------------------------------
 PER-TASK                                  score   median(s)     cv
------------------------------------------------------------------------
 Data prep & FE
   dp_groupby[pandas]                      0.92       1.41    0.01
   dp_groupby[polars]                      2.10       0.619   0.02
   fe_rolling[pandas]                      0.71       4.88    0.03
   fe_rolling[polars]                      3.40       1.02    0.02
   ...
 Linalg
   la_gemm                                 1.55       1.90    0.01
   ...
------------------------------------------------------------------------
 NOTES
   noisy (cv>0.10):   cl_gmm
   excluded:          (none skipped / failed / swapped)
   p/e isolation:     biased (QoS); mean off-core residency 6.2%
------------------------------------------------------------------------
 Comparable only across runs with matching version + mode + BLAS.
 JSON: results/2026-06-14T03-22-10Z_ab12cd.json
========================================================================
```

**Conditional sections:**
- **`--sweep`** adds a second integer column to the SCORE SUMMARY block (`1-core`) beside the
  all-cores column, plus a `scaling` line (all-cores ÷ single-core), so per-core quality and
  whole-machine throughput sit side by side without blending.
- **Heterogeneous chips** append an `E-CORE CONTRIBUTION` block: per-task `e_core_delta`
  (`>0 ⇒ E-cores help`) and the p-core overall — i.e. the E / all-cores pair. The full
  single-P / E / all-cores triplet is shown only when `--sweep` was also passed (which
  supplies the single-P-core leg); without it the block notes that single-P is absent.
- **No reference baseline yet:** the `score` column is dropped, `median(s)` becomes the only
  per-task number, and the SCORE SUMMARY shows `OVERALL  (no baseline — raw times only)` with
  the category lines omitted, rather than printing a faked score.
- **Non-standard `--threads N`** (neither 1 nor physical-all): same raw-times treatment — the
  SCORE SUMMARY header reads `OVERALL  (threads=N, non-standard — raw times only)` and the
  category/score lines are omitted, since no reference denominator exists for an arbitrary
  thread count. The thread count is shown in the identity header as usual.
- **`--summary`** prints only the identity header and SCORE SUMMARY block (the shareable core),
  omitting the per-task table and notes.

---

## 8. Scoring

SPEC-style, ratio-based, robust to one task — or one over-represented algorithm family —
dominating:

- **Per-task score** = `reference_median_s / measured_median_s` (so `>1` = faster than the
  reference machine, `<1` = slower). For data-prep / FE tasks that run on both engines, the
  pandas and Polars variants are **separate per-task scores** (`task[pandas]`, `task[polars]`),
  both feeding the `data_prep` category geomean — there is no single blended per-task number.
- **Category score** = geometric mean of that category's per-task scores.
- **Headline** = `100 × geometric_mean(the 6 category scores)` — a **category-weighted**
  geomean, so each category contributes equally regardless of how many tasks it holds (the
  boosting tasks no longer triple-count). The reference machine still scores exactly **100**.
- Computed independently for all-cores and single-core (and p-cores on heterogeneous chips);
  **never blended**. Only these canonical thread configs are scored, bucketed by the
  `(threads_mode, cores)` pair on each result: `threads_mode="all"` + `cores="all"` →
  `all_cores`; `threads_mode="single"` (i.e. `--threads 1`) → `single_core`;
  `threads_mode="all"` + `cores="p"` (the heterogeneous auto-expanded pass) → `p_cores`;
  `--sweep` → both `all_cores` and `single_core`. A **non-standard `--threads N`**
  (`threads_mode="explicit"`, neither 1 nor physical-all) has no reference denominator, so it
  is **not scored** — the run emits raw times + intra-run ratios only and the report labels it
  `threads=N (non-standard — raw times only)`.
- **Robustness guards.** Results with `status != "ok"` or `swapped: true` are excluded; if
  exclusions leave a category empty, that category drops out of the headline and the report
  says so. The geomean is also guarded against a zero/negative per-task score (clamped /
  reported as an error rather than silently producing `0` or `NaN`). On homogeneous CPUs,
  `p_cores` and `e_core_delta` are omitted entirely.

### Reference baseline
`baselines/reference.json` is a checked-in artifact produced on a documented, **reproducible**
reference machine. Key points:

- **The all-cores score is the headline; single-core is the portable per-core property.** The
  all-cores score is a property of the CPU *and its core count* relative to the reference: a
  per-task all-cores ratio is `ref_all / your_all`, so more cores → higher score by
  construction. That is the intended whole-machine meaning and is exactly the number we want to
  lead with. The single-core score is the figure to consult when you need to factor core count
  out and compare per-core quality; the report presents it as the secondary diagnostic.
- **`reference.json` stores two per-task baseline sets:** `single_core` medians and
  `all_cores` medians (the latter captured at the reference machine's own physical-core
  count). On the test machine:
  - `single_core` score = `ref_single / test_single`.
  - `all_cores` score = `ref_all / test_all`.
  - `p_cores` **reuses the `all_cores` reference denominator**, so p-cores and all-cores sit
    on one comparable scale and the gap between them is the E-core contribution in score
    units. (The reference is homogeneous and has no separate "p-cores" run.)
  - `e_core_delta` uses **no reference** — it is the intra-machine time ratio defined in §2.4.
- **Reference machine — dev placeholder, cloud SKU later.** *During development* the reference
  is the **developer's own machine**: run the suite with `--sweep`, extract the baseline, and
  commit it so the scoring/report code lights up end-to-end. This baseline is a **placeholder,
  not for publishing comparisons** (the dev machine scores ~100 against itself by construction).
  Before release it is **replaced** by a baseline produced on a publicly-rentable cloud SKU (a
  current general-purpose x86 instance). Either way the file records the machine identity (instance
  type or host spec), core topology, OS image, BLAS backend, and resolved lockfile hash; every
  results file is stamped with `reference_version`, and `compare` refuses across reference
  versions just as it does across `benchmark_version` — so the dev→cloud swap is a clean,
  detectable `reference_version` bump.
- **Bootstrapping.** Until the file exists, the report shows raw times and intra-run ratios
  only, and the headline is omitted rather than faked.
- **How it's produced.** `reference.json` is not hand-written: run the full suite with
  `--sweep` on the chosen reference SKU, then extract the per-task `single_core` and `all_cores`
  medians from that results JSON into the baseline file (a small `cpubench` helper does the
  extraction and stamps `reference_version` + machine spec). It is then committed.

**Canonical categories (fixed order).** The headline geomean and every report iterate the six
categories in this exact order: `data_prep`, `linalg`, `factorization`, `clustering`, `models`,
`sparse`. Display labels are *Data prep & FE*, *Linalg*, *Factorization*, *Clustering*,
*Models*, *Sparse / NLP*. This ordering is fixed so two text reports align line-for-line.

---

## 9. Repository layout

```
cpubench/
├── README.md
├── SPEC.md
├── pyproject.toml          # deps + optional groups, build config
├── uv.lock                 # universal cross-platform lock
├── cpubench/
│   ├── __init__.py
│   ├── cli.py              # argument parsing, orchestration
│   ├── controller.py       # spawns workers, collects result files, --resume
│   ├── worker.py           # runs ONE (task,size,threads,cores) config in isolation
│   ├── runner.py           # warm-up + fixed repetition + timing + RSS
│   ├── registry.py         # @task decorator, deterministic ordering, mode resolution
│   ├── environment.py      # CPU/RAM/OS/Python/BLAS/P-E detection
│   ├── threading_ctl.py    # env-var + threadpoolctl + per-lib thread wiring
│   ├── affinity.py         # P/E detection + modular per-platform enforcement backend
│   │                       #   (Linux/Win hard pin · macOS QoS via taskpolicy) +
│   │                       #   single-thread P-core placement, per-core residency sampling
│   ├── memory.py           # peak-RSS tracking, swap detection
│   ├── datasets.py         # seeded synthetic generators (frames, matrices, ML sets)
│   ├── scoring.py          # ratios, geomeans, e_core_delta, baseline loading + extraction
│   ├── reporting.py        # txt (canonical) + rich console + md/html
│   └── tasks/
│       ├── data_prep.py    # core dp_ + feature-engineering fe_ tasks; pandas + polars variants
│       ├── linalg.py
│       ├── factorization.py
│       ├── clustering.py
│       ├── models.py
│       └── sparse.py
├── baselines/
│   └── reference.json      # reference machine scores (dev placeholder, then cloud SKU)
├── results/                # run outputs (gitignored); .partial/ holds per-config result files
└── tests/                  # correctness smoke + harness unit tests
```

### Task registration pattern
```python
@task(name="la_gemm", category="linalg",
      sizes={"quick": {"N": 2000}, "normal": {"N": 8000}},
      modes={"quick", "normal"},          # heavier tasks omit "quick"
      backend_sensitive=True)             # default False; True for linalg/factorization + md_gpr
def la_gemm(params, ctx):
    A, B = ctx.data                # generated untimed
    with ctx.timer():              # only this is measured
        C = A @ B
    return {"trace": float(C.trace())}   # optional, informational checksum (§3)
```

The **registry is the single source of truth** for the task list, sizes, and modes — there are
no separate `configs/*.yaml` files. `cpubench list` renders the resolved tasks/categories/sizes
straight from the registry.

The `ctx` handed to a task provides: `ctx.data` (the pre-generated, untimed inputs), a
`ctx.timer()` context manager wrapping the one timed region, `ctx.params` (resolved sizes),
`ctx.threads` (the thread count to pass to `n_jobs` / `num_threads`), and `ctx.rng`
(`default_rng(seed)`) for any in-task randomness. The decorator also accepts `engines=("pandas",
"polars")` for data-prep / FE tasks — the harness then runs the task once per engine and records
each as a separate scored entry (`name[engine]`). Long-runners simply omit `"quick"` from
`modes`.

---

## 10. Dependencies & environment (`uv`)

`pyproject.toml` dependency groups:

- **core (required):** `numpy`, `scipy`, `scikit-learn`, `pandas`, `polars`, `lightgbm`,
  `threadpoolctl`, `psutil`, `rich`.
- **dev:** `pytest`, `ruff`, formatter.

Reproducibility:
- `uv lock` produces a **universal lockfile** covering Linux / macOS / Windows and
  x86_64 / arm64.
- Pin a single supported Python (**3.12**); CI runs the matrix
  {Linux, macOS-arm64, Windows} × {3.12}.
- Run: `uv sync` then `uv run cpubench run`.

> **Lock now resolves easily.** Removing the UMAP dependency dropped `numba` + `llvmlite`,
> which were the binding constraint on the NumPy/Python range. Without them the resolution is
> far less likely to conflict and NumPy can track a current release. Python 3.12 is comfortably
> within every remaining dependency's support window.

> **Install caveats to verify during scaffolding:**
> - LightGBM needs an OpenMP runtime; on macOS this is `libomp`. Modern wheels generally
>   bundle/declare it, but the README should document `brew install libomp` as a fallback.
> - Confirm `environment.blas_backend` detection resolves **Accelerate** correctly on
>   macOS-arm64 (it is the finickiest backend to identify and the most consequential for
>   linalg scores). Use `numpy.show_config()` + `threadpoolctl` and verify against a known box.

---

## 11. Correctness vs timing
This is a **timing** benchmark, not a numerical-accuracy test. Results across thread counts
may differ in the last bits (parallel reductions, random init); that is acceptable. Two
guards keep the timings *meaningful*:

- A small `tests/` smoke suite asserts each task *runs and returns the expected output
  shape/dtype* in `quick` mode, so a broken task fails loudly instead of posting a fast time.
  **This is the primary correctness guard.**
- A task *may* return a cheap summary value (inertia, loss, trace, output hash) for sanity, but
  it is **informational only** — not validated against a golden value, not tolerance-calibrated,
  and not a `compare` gate (§3, §6). The goal is high-level speed measurement, not numerical
  validation.

To keep the workload itself stable across library versions:
- Synthetic data generators use **`numpy.random.default_rng(seed)`** directly rather than
  sklearn's `make_*` RNGs, so a scikit-learn upgrade can't silently change the data.
- Generators **pin explicit dtypes** (`np.int64`, `np.float64`, etc.) on every array they
  produce — never the default integer dtype, which is the C `long` and is **int32 on Windows
  but int64 on Linux/macOS**. With dtypes pinned and the numpy version locked, the PCG64 stream
  and its uniform/integer draws are byte-identical across architectures (x86_64, arm64), so the
  generated data is the same everywhere. (The remaining cross-architecture variation — last-bit
  differences in transcendental distribution tails, and convergence/tie decisions inside
  iterative estimators that depend on the BLAS backend — is accepted; it is small and not
  corrected for.)
- scikit-learn and LightGBM estimators are seeded with **`random_state = seed` (1337)**, so
  tree structure, initialization, and solver paths — and therefore the work being timed — are
  reproducible.

---

## 12. Known caveats (surfaced in the report)
1. The all-cores score is the intended headline and scales with core count — it is a
   whole-machine throughput number, by design. To compare per-core quality with core count
   factored out, consult the single-core score (under `--sweep`); the two are never blended.
2. Apple Silicon P/E asymmetry depresses naive scaling. True P-core isolation is **not
   enforceable on macOS** (QoS bias only, verified by measured off-core residency); evaluate
   from the single-P-core / E-core / all-cores triplet and the per-task E-core delta, not a
   single P-only number.
3. Thermal throttling is **intentionally retained** as real-world system behaviour and is not
   flagged or corrected for; results reflect the machine as it actually performs under load.
   Task order is deterministic, so the *sequence* is the same run-to-run, but on thermally
   limited machines (e.g. fanless laptops) absolute times still depend on ambient conditions
   and position in the run. The 2 s inter-task buffer is only a settling gap, not a thermal
   reset.
4. BLAS backend differences (Accelerate vs OpenBLAS vs MKL) can swing linalg results 2–3×;
   the backend is always reported and cross-backend comparisons are labelled accordingly.
5. pandas (mostly single-threaded) vs Polars (multi-threaded) is intentionally compared
   as-shipped at all-cores, and pinned-equal at `--threads 1`.
6. The default uses **physical** cores, not logical, to report the machine's best *honest*
   whole-system throughput. For compute-bound work, physical is peak throughput — adding the
   SMT/hyperthreading siblings would *reduce* it, so logical would be a deoptimization dressed
   up as "the whole machine". The trade is that a few latency-bound parallel tasks could gain
   slightly from SMT; that upside is reachable via explicit `--threads N` or a sweep, which is
   also how to probe where it actually exists.

---

## 13. Resolved decisions & remaining scaffold confirmations

Resolved:
- **Python version:** 3.12, single supported version (confirmed within all dependency support
  windows now that numba/llvmlite are gone).
- **`dr_umap` / dimensionality reduction:** removed entirely. The benchmark now has 6
  categories; quick and normal cover the same set.
- **Repetitions:** fixed `repeat` (default 5). Adaptive repetition and `time_budget_per_task`
  removed. The only time limit is a **per-config hang ceiling `--timeout` (default 3600 s = 1 h,
  `0` disables)** — a safety net against a hung/pathologically-slow worker, not a per-task
  budget; reps run to completion inside it. Timeout → `status: "failed"`.
- **Warm-up:** default ON.
- **Default thread count:** physical cores.
- **Memory:** **no pre-task estimate guard and no `mem_estimate`** — `peak_rss_mb` is recorded
  per config and sizes are tuned down to target machines from those measurements (normal peak
  target ~8 GB; target machines assumed > 16 GB RAM). A too-big task OOM-fails (`status:
  "failed"`); swap during a task flags `swapped: true` and excludes it from scoring.
- **P/E detection:** best-effort; **uncertain or unavailable detection ⇒ treat as homogeneous**
  (no guessed split), and the heterogeneous-only blocks are omitted.
- **macOS QoS:** set via the `taskpolicy(8)` CLI at worker-spawn (`-b` for `--cores e`); P-core
  isolation has no shell lever and is reported as `biased`, verified by 10 Hz off-core residency
  sampling (`>10%` ⇒ flagged leaky).
- **Checksums:** **informational only** — no tolerance calibration, not a `compare` gate.
  Correctness is guarded by the `tests/` shape/dtype smoke suite.
- **pandas/Polars FE:** same *logical* feature, but **exact cross-engine equivalence is not
  required or asserted**; the two are scored separately.
- **Config source of truth:** the **registry**; no `configs/*.yaml` files.
- **Non-standard `--threads N`** (neither 1 nor physical-all): produces raw times + intra-run
  ratios only, no scored bucket, labelled as such in the report.
- **`md_lgbm_multi`:** `num_class=10`, 250 rounds, `normal` only.
- **Feature engineering:** folded into the `data_prep` category (still **6 categories**), not a
  standalone 7th — avoids one workload family carrying outsized headline weight. Added as
  distinct FE tasks (`fe_lags`, `fe_rolling`, `fe_expanding`, `fe_ewm`, `fe_onehot`,
  `fe_rank`, `fe_datetime`) rather than one end-to-end pipeline. Panel `normal` bounded to 10M
  rows / 20k entities.
- **RF inference:** added `md_rf_predict` to `models` (train untimed, predict timed).
- **Reporting:** category scores are a first-class block shown beneath the headline.
- **Versioning:** these additions bump `benchmark_version` to **1.1.0**; `compare` will refuse
  across the bump, and `baselines/reference.json` must be regenerated to include the new tasks
  before the headline is valid again.
- **Seeds:** data `default_rng(1337)`; estimator `random_state=1337`.

To confirm at scaffold time:
- **Reference baseline:** for now produced on the **developer's own machine** as a placeholder
  so scoring lights up; pick and document the publicly-rentable **cloud SKU** and regenerate the
  baseline (a `reference_version` bump) before release.
- Verify the `uv` lock resolves on all three OSes × {x86_64, arm64} under Python 3.12.
- Verify `libomp` availability for LightGBM on macOS-arm64 wheels (document the `brew` fallback
  if needed) and confirm Accelerate backend detection.

Open issues (raised, not yet resolved):
- **macOS P-core isolation is soft.** `taskpolicy -b` confines E-core (`--cores e`) work
  reliably, but there is no shell/Python lever that *forces* P-cores, so `--cores p` on macOS
  stays best-effort (`biased`), trusting the residency flag to mark leaky runs. Accepted **as-is
  for now**; `affinity.py` keeps the enforcement strategy **modular** (a per-platform
  pin/QoS/verify backend) so a better macOS P-core mechanism can drop in later without touching
  the runner or scoring. Deferred, not blocking.

---

## 14. Implementation handoff

Suggested build order (each layer is testable before the next):

1. **`pyproject.toml` + `uv.lock`** (Python 3.12, the core deps from §10), then `environment.py`
   (CPU/RAM/OS/Python/BLAS + P/E detection) and `cpubench env`.
2. **`threading_ctl.py`** and **`affinity.py`** (pre-import env vars, threadpoolctl, single-thread
   P-core placement, hard pin vs QoS bias, residency sampling).
3. **`registry.py`** (`@task`, deterministic ordering, modes, `engines`, `backend_sensitive`)
   + **`datasets.py`** (seeded generators, **pinned dtypes**, the shared frame/panel built once
   and handed to both engines).
4. **`worker.py`** + **`runner.py`** (subprocess, warm-up, fixed reps, no-reset/read-only
   discipline, file-based result protocol, `peak_rss_mb` + swap sampling) and **`memory.py`**
   (peak-RSS tracking, swap detection), then **`controller.py`** (spawn, `--timeout` kill,
   collect, `--resume`).
5. **`tasks/`** — implement the catalogue (§5) one category at a time, each with a `quick`-mode
   smoke test asserting shape/dtype (dual-engine tasks run on both engines but are **not**
   asserted equal across engines).
6. **`scoring.py`** (ratios, category geomeans in the fixed order, `e_core_delta`, baseline
   load + extraction) and **`reporting.py`** (the §7.3 txt report first, then `rich`/md/html).

**Deliberately deferred — do not block on these, and do not fake them:**
- `baselines/reference.json` does not exist yet. Build the scoring code; until a baseline exists
  the report shows raw times + intra-run ratios and **omits the headline** (§8). A **dev-machine
  placeholder** baseline is produced first to exercise scoring end-to-end; it is replaced by the
  cloud-SKU baseline (a `reference_version` bump) before release.
- Several sizes (FE panel, `md_gpr`, `md_rf_predict`, `sp_fhash`) are first guesses. Measure real
  `peak_rss_mb` / time on first runs and tune sizes down to the target machines ("measure then
  tune"). There is no `mem_estimate` to maintain — observed RSS is the tuning signal.

**Definition of done for the handoff:** `uv run cpubench run --mode quick` completes on
Linux/macOS/Windows, streams resumable per-config JSON, emits a valid schema-v1 results file and
the §7.3 txt report (raw-times mode, no headline), and the `tests/` smoke suite (shape/dtype)
passes. The headline lights up once the dev-machine placeholder baseline is committed, and is
re-based on the cloud SKU before release.