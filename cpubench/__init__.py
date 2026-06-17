"""ml-cpu-bench: reproducible CPU benchmark for classical ML + data-science workloads."""

BENCHMARK_VERSION = "1.1.0"
SCHEMA_VERSION = 1
DEFAULT_SEED = 1337

# Fixed category order (SPEC §8). Headline geomean and every report iterate in this order.
CATEGORY_ORDER = (
    "data_prep",
    "linalg",
    "factorization",
    "clustering",
    "models",
    "sparse",
)

CATEGORY_LABELS = {
    "data_prep": "Data prep & FE",
    "linalg": "Linalg",
    "factorization": "Factorization",
    "clustering": "Clustering",
    "models": "Models",
    "sparse": "Sparse / NLP",
}
