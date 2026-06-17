# ml-cpu-bench

A reproducible benchmark that measures **CPU** performance on classical machine-learning and
data-science workloads — no neural nets, no GPU. Clone, run one command, and get per-task and
per-category numbers for how good a given CPU/system is at real ML + data-prep work.

It exercises the parts of the stack that classical ML actually stresses: dense BLAS/LAPACK,
scikit-learn estimators, LightGBM, sparse/text pipelines, and pandas-vs-Polars data wrangling —
all on synthetic, fixed-seed data so results are stable across machines and runs.

See [`SPEC.md`](SPEC.md) for the authoritative design and [`CLAUDE.md`](CLAUDE.md) for the quick
reference and the invariants that keep the numbers honest.

---

## Requirements

The **only** prerequisite is [`uv`](https://docs.astral.sh/uv/). You do **not** need conda, a
manual virtualenv, or a pre-installed Python.

```bash
# install uv (pick one)
curl -LsSf https://astral.sh/uv/install.sh | sh     # Linux / macOS
brew install uv                                      # macOS (Homebrew)
```

- **Python 3.12** is required, but `uv` downloads and manages it for you if the machine doesn't
  have one — `pyproject.toml` pins `requires-python = "==3.12.*"`.
- Dependencies are pinned via a **universal `uv.lock`** (Linux / macOS / Windows × x86_64 /
  arm64), so `uv sync` reproduces the exact same versions everywhere.
- **macOS only:** LightGBM needs an OpenMP runtime. Modern wheels usually bundle it; if
  `import lightgbm` fails, run `brew install libomp`.

---

## Quick start

```bash
git clone <repo> && cd ml-cpu-bench
uv sync                              # creates .venv/ and installs the locked deps
uv run cpubench run --mode quick     # fast subset — the dev/CI loop
```

That's it — `uv sync` builds a project-local `.venv/`, and `uv run` activates it per command, so
there's no separate "activate the environment" step. Both `.venv/` and `results/` are gitignored
and regenerate on a fresh clone.

### Commands

```bash
uv run cpubench run --mode quick     # fast subset (excludes the long-runners)
uv run cpubench run                  # full normal-mode run (every task)
uv run cpubench run --sweep          # add a single-core pass (per-core diagnostic)
uv run cpubench list                 # list tasks + categories + sizes (from the registry)
uv run cpubench env                  # print detected CPU / RAM / cores / BLAS / P-E only
uv run cpubench report FILE          # re-render a report from a results JSON
uv run cpubench compare A B          # diff two runs (noise- and version-aware)
```

Useful `run` flags: `--threads N` / `--cores all|p|e`, `--tasks t1,t2` / `--exclude t1`,
`--repeat N` (default 5), `--seed S` (default 1337), `--timeout SECONDS` (per-config hang
ceiling, default 3600, `0` disables), `--cooldown SECONDS` (default 2), `--no-warmup`,
`--resume`, `--out PATH`, `--format txt|md|html`, `--summary`, `--no-report`.

---

## What it measures (6 categories)

| Category | Examples | What it stresses |
|---|---|---|
| **data_prep** | `dp_groupby/join/sort/filter/string/rolling`, `fe_lags/rolling/expanding/ewm/onehot/rank/datetime` | DataFrame engines — run on **both pandas and Polars**, scored separately |
| **linalg** | `la_gemm/solve/cholesky/qr/svd/eigh/fft` | dense BLAS/LAPACK (backend-sensitive) |
| **factorization** | `mf_tsvd/pca/nmf` | matrix decomposition (backend-sensitive) |
| **clustering** | `cl_kmeans/mbkmeans/optics/gmm` | distance/density/EM workloads |
| **models** | `md_linreg/ridge/lasso/logreg/bayes_ridge/gpr/rf/rf_predict/hist_gbm/lgbm/lgbm_multi/svc_rbf/knn` | scikit-learn + LightGBM fit/predict |
| **sparse** | `sp_tfidf/hashvec/fhash/matmul/tsvd/nmf/saga`, `nlp_lda`, `sp_lasso_cv` | sparse + NLP/text pipelines (backend-neutral) |

Two run **modes**: `quick` (a fast subset for CI/dev, excludes the long-runners `md_gpr`,
`md_lgbm_multi`, `nlp_lda`, `sp_lasso_cv`) and `normal` (every task, the real benchmark).

### Task reference

Naming: the prefix is the category (`dp_`/`fe_` data-prep, `la_` linalg, `mf_` factorization,
`cl_` clustering, `md_` models, `sp_`/`nlp_` sparse). 🔁 = run on **both pandas and Polars**
(scored separately); ⚙ = **backend-sensitive** (linalg/factorization BLAS backend matters);
🕒 = **`normal`-mode only** (excluded from `quick`). `cpubench list` prints the live list + sizes.

**Data prep & feature engineering** — DataFrame throughput, every task on both engines 🔁

| Task | What it does |
|---|---|
| `dp_groupby` | group by a categorical, aggregate sum/mean/std/count over 4 numeric columns |
| `dp_join` | inner join two frames on a key (controlled ~1:1 cardinality, no blow-up) |
| `dp_sort` | multi-key sort on 2 columns (non-mutating) |
| `dp_filter` | boolean-mask filter + column selection |
| `dp_string` | regex extract + lowercase + substring-contains on a string column |
| `dp_rolling` | global rolling window (mean/std/median, window=100) over the time-sorted frame |
| `fe_lags` | per-entity lag features — grouped `shift` by 1 / 7 / 30 |
| `fe_rolling` | per-entity **leakage-safe** rolling means (7/30/90) + std(30), prior obs only |
| `fe_expanding` | per-entity expanding mean + historical normalization (value ÷ mean of prior obs) |
| `fe_ewm` | per-entity exponentially-weighted moving average |
| `fe_onehot` | one-hot encode the low/medium-cardinality categoricals (output `uint8`) |
| `fe_rank` | cross-sectional rank + z-score per timestamp (grouping across entities) |
| `fe_datetime` | calendar features (day-of-week/month/quarter) + cyclical sin/cos + vectorized `log1p`/`clip` |

**Linalg** — dense BLAS/LAPACK stress, all ⚙

| Task | What it does |
|---|---|
| `la_gemm` | dense matrix multiply — **ge**neral **m**atrix-**m**atrix product (`A @ B`) |
| `la_solve` | solve a dense linear system `Ax = b` (LU factorization) |
| `la_cholesky` | Cholesky factorization of a symmetric positive-definite matrix |
| `la_qr` | QR decomposition of a tall matrix |
| `la_svd` | singular value decomposition (thin, `full_matrices=False`) |
| `la_eigh` | symmetric eigendecomposition (eigenvalues of a symmetric matrix) |
| `la_fft` | 1-D complex fast Fourier transform |

**Factorization** — matrix decomposition via scikit-learn, all ⚙

| Task | What it does |
|---|---|
| `mf_tsvd` | truncated SVD on a dense matrix (`TruncatedSVD`, k=50) |
| `mf_pca` | full principal-component analysis (`PCA`, k=50) |
| `mf_nmf` | non-negative matrix factorization on a dense matrix (k=20) |

**Clustering** — scikit-learn

| Task | What it does |
|---|---|
| `cl_kmeans` | KMeans (Lloyd's algorithm) |
| `cl_mbkmeans` | MiniBatchKMeans (mini-batch KMeans, batch=1024) |
| `cl_optics` | OPTICS density-based clustering — full reachability ordering, ball-tree, bounded `max_eps` |
| `cl_gmm` | Gaussian Mixture Model, full covariance (EM) |

**Models** — scikit-learn + LightGBM fit/predict

| Task | What it does |
|---|---|
| `md_linreg` | ordinary least-squares linear regression |
| `md_ridge` | ridge (L2) regression with built-in cross-validation (`RidgeCV`) |
| `md_lasso` | lasso (L1) regression with CV (coordinate descent, `LassoCV`) |
| `md_logreg` | multinomial logistic regression (lbfgs, 3-class) |
| `md_bayes_ridge` | Bayesian ridge regression |
| `md_gpr` 🕒 ⚙ | Gaussian-process regression, fixed RBF kernel — O(n³) dense kernel Cholesky |
| `md_rf` | RandomForest classifier — **training** |
| `md_rf_predict` | RandomForest **inference** — forest trained untimed, only `predict` on a large set is timed |
| `md_hist_gbm` | histogram-based gradient-boosted trees (`HistGradientBoostingClassifier`) |
| `md_lgbm` | LightGBM gradient boosting — regression |
| `md_lgbm_multi` 🕒 | LightGBM multiclass softmax (`num_class=10`, ~250 rounds) |
| `md_svc_rbf` | support-vector classifier, RBF kernel (single-threaded dominator, kept small) |
| `md_knn` | k-nearest-neighbours classifier (brute force, 1000 queries) |

**Sparse / NLP** — sparse + text pipelines (backend-neutral)

| Task | What it does |
|---|---|
| `sp_tfidf` | `TfidfVectorizer.fit_transform` on a synthetic Zipfian token corpus |
| `sp_hashvec` | `HashingVectorizer.transform` — the hashing trick (stateless, single-threaded) |
| `sp_fhash` | `FeatureHasher` on high-cardinality `field=value` strings (hashing trick) |
| `sp_matmul` | sparse CSR × dense matrix product (SpMM primitive) |
| `sp_tsvd` | randomized truncated SVD on a sparse matrix (LSA use case) |
| `sp_nmf` | NMF on a sparse TF-IDF doc-term matrix (topic model — distinct sparse code path) |
| `sp_saga` | logistic regression with the SAGA solver on sparse input (binary) |
| `nlp_lda` 🕒 | Latent Dirichlet Allocation topic model (variational Bayes) |
| `sp_lasso_cv` 🕒 | `LassoCV` on sparse regression data (CV folds multiply the work) |

---

## How it works

```
cli → controller → (per config) worker subprocess → runner (timed reps) → result file
                 → collect partial files → scoring → reporting
```

Each `(task, size, threads, cores)` config runs in its **own subprocess**, which is what makes
the numbers trustworthy:

- **Honest timing.** Only the algorithm under test is inside the timed region — data generation,
  imports, allocation, and frame construction are untimed. One discarded warm-up run, then a
  fixed `repeat` (default 5) timed reps; the timed region is read-only (no reset between reps).
- **Fair threading.** A single `--threads` value is honoured by *all* parallel libraries (BLAS,
  scikit-learn, LightGBM, Polars, numexpr) via env vars set **before any heavy import**. Default
  is **physical** cores.
- **Reproducible data.** Everything is synthetic via `numpy.random.default_rng(1337)` with
  explicitly pinned dtypes (`int64`/`float64`/`uint8`), so the generated bytes are identical
  across OS/arch. Estimators are seeded with `random_state=1337`.
- **Resumable.** Per-config partial files (`results/.partial/<id>.json`) are the incremental
  stream and the `--resume` source of truth; the final schema-v1 JSON is assembled from them.
- **Memory- and swap-aware.** Each config records `peak_rss_mb`; a too-big task OOM-fails
  (`status: "failed"`, excluded) rather than hanging the suite, and any task that swaps is
  flagged and excluded from scoring.

Source modules live in [`cpubench/`](cpubench/): `environment.py`, `threading_ctl.py`,
`affinity.py`, `registry.py`, `datasets.py`, `worker.py`, `runner.py`, `memory.py`,
`controller.py`, `scoring.py`, `reporting.py`, and the per-category task modules in
[`cpubench/tasks/`](cpubench/tasks/). The **registry is the single source of truth** for the
task list, sizes, and modes (no config files).

---

## Output

Every run writes two files (default into `results/<run_id>.{json,txt}`):

- An **authoritative JSON** (`schema_version: 1`) with full per-config detail: all raw reps,
  median/min/std/cv, `peak_rss_mb`, `cpu_wall_ratio`, checksum, status, and the detected
  environment.
- A **plain-text report** (§7.3): pure ASCII, ≤90 columns, deterministic ordering — built to
  paste into an issue/gist and `diff` cleanly. Each per-task row shows the **score next to its
  timing** (median/min/cv) and peak RSS. `--format md|html` emits those too.

### Scores are deferred (raw-times mode)

Scores are ratios against a reference machine (`reference_machine = 100`), but
**`baselines/reference.json` is intentionally not committed yet**. Until it is, every run shows
**raw times and intra-run ratios only**, and the headline is omitted
(`OVERALL (no baseline -- raw times only)`). This is by design, not a setup gap — so the numbers
in a freshly cloned run are this machine's raw medians, not cross-machine scores. The scoring and
report code is fully written and unit-tested against an in-repo fixture baseline; the headline
lights up once a baseline is added.

### Per-task score (once a baseline exists)

`score = reference_median_s / measured_median_s` (`>1` = faster than reference). Category score =
geomean of its per-task scores; **headline = `100 × geomean of the 6 category scores`**
(category-weighted). pandas and Polars are separate scored entries. The **all-cores** score is
the headline (whole-machine throughput); **single-core** (under `--sweep`) is the secondary
per-core diagnostic. The two are never blended.

---

## Development

```bash
uv run pytest                        # shape/dtype smoke suite + harness/scoring/report units
uv run ruff check . && uv run ruff format .
```

The `tests/` smoke suite is the **primary correctness guard**: it asserts every task runs in
quick mode and returns the expected output shape/dtype (dual-engine tasks run on both engines but
are not asserted equal across engines). Build/test one category at a time; commit in small steps.

---

## Notes & caveats

- **Physical, not logical, cores by default** — for compute-bound work that's peak honest
  throughput; SMT siblings add contention, not FP units. Probe SMT explicitly with `--threads N`.
- **BLAS backend matters.** Accelerate vs OpenBLAS vs MKL can swing linalg results 2–3×; the
  backend is always detected and reported, and cross-backend comparisons are labelled.
- **Apple Silicon P/E.** True P-core isolation isn't enforceable on macOS (QoS bias only, with
  measured off-core residency); evaluate from the single-P / E-core / all-cores triplet. On
  homogeneous chips the P/E sections are omitted entirely.
- **Thermal throttling is retained** as real-world behaviour, not corrected for.
- **Checksums are informational only** — not validated, not a `compare` gate.
- On an 8 GB machine, several `normal`-mode tasks will OOM-fail by design (sizes target
  ≤ ~8 GB RSS assuming > 16 GB RAM); **quick mode is the dev loop.**
