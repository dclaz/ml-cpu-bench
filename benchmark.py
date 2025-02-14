from datetime import datetime
import json
import argparse
from benchmarks.benchmarks import *
from benchmarks.system import get_system_info, get_package_versions

import numpy as np
import random

# Set all possible seeds
np.random.seed(42)
random.seed(42)


def main():
    parser = argparse.ArgumentParser(description="CPU Benchmarking Script")
    parser.add_argument(
        "--n_runs", type=int, default=3, help="Number of runs for each benchmark"
    )
    parser.add_argument(
        "--output", type=str, default="benchmark_results.txt", help="Output file name"
    )
    args = parser.parse_args()

    system_info = get_system_info()
    package_versions = get_package_versions()

    benchmarks = {
        "matrix_multiplication": matrix_multiplication_benchmark(),
        "matrix_inversion_scipy": matrix_inversion_benchmark_scipy(),
        "matrix_inversion_numpy": matrix_inversion_benchmark_numpy(),
        "pca": pca_benchmark(),
        "svd": svd_benchmark(),
        "nmf": nmf_benchmark(),
        "kmeans": kmeans_benchmark(),
        "umap": umap_benchmark(),
        "pandas_moving_average": pandas_moving_average_benchmark(),
        "pandas_aggregation": pandas_aggregation_benchmark(),
        "lasso_cv": lasso_cv_benchmark(),
        "sgd_classifier": sgd_classifier_benchmark(),
        "lgbm": lgbm_benchmark(),
    }

    results = {}
    for name, benchmark in benchmarks.items():
        results.update(run_benchmark(benchmark, name, args.n_runs))

    # Save results to file
    output_data = {
        "system_info": system_info,
        "package_versions": package_versions,
        "benchmark_results": results,
        "timestamp": datetime.now().isoformat(),
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=4)


if __name__ == "__main__":
    main()
