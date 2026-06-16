# ml-cpu-bench

A reproducible benchmark measuring **CPU** performance on classical ML and
data-science workloads (no neural nets, no GPU). Clone, run one command, get
per-task and category scores.

See [`SPEC.md`](SPEC.md) for the authoritative design and [`CLAUDE.md`](CLAUDE.md)
for the quick reference and invariants.

## Quick start

```bash
uv sync                              # install pinned deps (Python 3.12)
uv run cpubench run --mode quick     # fast subset — the dev/CI loop
uv run cpubench run                  # full normal-mode run
uv run cpubench run --sweep          # add single-core pass (per-core scores)
uv run cpubench list                 # tasks + categories + sizes
uv run cpubench env                  # detected CPU/RAM/BLAS/P-E only
uv run cpubench report FILE          # re-render a report from results JSON
uv run cpubench compare A B          # diff two runs (noise/version aware)
```

## Notes

- Python **3.12** only. Dependencies are pinned via a universal `uv.lock`.
- **LightGBM** needs an OpenMP runtime. Modern wheels generally bundle it; on
  macOS the fallback is `brew install libomp`.
- All data is synthetic and fixed-seed; no disk I/O is timed.
