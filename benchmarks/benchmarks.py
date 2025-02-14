import argparse
import json
import time
from datetime import datetime

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import scipy
import scipy.linalg
import sklearn.metrics
import sklearn.model_selection
import umap
from optuna.samplers import TPESampler
from sklearn.cluster import KMeans
from sklearn.datasets import make_classification
from sklearn.decomposition import NMF, PCA
from sklearn.linear_model import LassoCV, SGDClassifier


def time_task(func, *args):
    """Times a given task."""
    start_time = time.time()
    func(*args)
    end_time = time.time()
    return end_time - start_time


def run_benchmark(task, task_name, n_runs=1):
    """Runs a benchmark task N times and returns timing statistics."""
    timings = []
    for _ in range(n_runs):
        timings.append(time_task(task))

    timings = np.array(timings)
    results = {
        "n_runs": n_runs,
        "average": np.mean(timings),
        "median": np.median(timings),
        "25th_percentile": np.percentile(timings, 25),
        "75th_percentile": np.percentile(timings, 75),
        "std_dev": np.std(timings),
    }
    print(f"Task: {task_name}, Results: {results}")
    return {task_name: results}

####################################################################################################
# Actual benchmarks below
####################################################################################################

def matrix_multiplication_benchmark(matrix_size=5000):
    """Benchmarks matrix multiplication."""
    A = np.random.rand(matrix_size, matrix_size)
    B = np.random.rand(matrix_size, matrix_size)

    def task():
        A @ B

    return task


def matrix_inversion_benchmark_scipy(matrix_size=5000):
    """Benchmarks matrix inversion using scipy."""
    A = np.random.rand(matrix_size, matrix_size)
    matrix = A @ A.T  # Ensure matrix is invertible

    def task():
        scipy.linalg.inv(matrix)

    return task


def matrix_inversion_benchmark_numpy(matrix_size=5000):
    """Benchmarks matrix inversion using numpy."""
    A = np.random.rand(matrix_size, matrix_size)
    matrix = A @ A.T  # Ensure matrix is invertible

    def task():
        np.linalg.inv(matrix)

    return task


def pca_benchmark(n_components=100, n_samples=100000, n_features=500):
    """Benchmarks PCA."""
    data = np.random.rand(n_samples, n_features)

    def task():
        pca = PCA(n_components=n_components)
        pca.fit(data)

    return task


def svd_benchmark(matrix_size=5000):
    """Benchmarks SVD."""
    matrix = np.random.rand(matrix_size, matrix_size)

    def task():
        scipy.linalg.svd(matrix)

    return task


def nmf_benchmark(n_components=50, n_samples=100000, n_features=200):
    """Benchmarks Non-Negative Matrix Factorization."""
    data = np.random.rand(n_samples, n_features)
    data = np.abs(data)  # NMF requires non-negative data

    def task():
        nmf = NMF(n_components=n_components, init="nndsvda", max_iter=200)
        nmf.fit(data)

    return task


def kmeans_benchmark(n_clusters=8, n_samples=100000, n_features=100):
    """Benchmarks K-means clustering."""
    data = np.random.rand(n_samples, n_features)

    def task():
        kmeans = KMeans(
            n_clusters=n_clusters,
            n_init=2,
        )
        kmeans.fit(data)

    return task


def umap_benchmark(n_neighbors=15, n_components=2, n_samples=100000, n_features=100):
    """Benchmarks UMAP."""
    data = np.random.rand(n_samples, n_features)

    def task():
        umap_reducer = umap.UMAP(
            n_neighbors=n_neighbors, n_components=n_components, n_jobs=-1, parallel=True
        )
        umap_reducer.fit(data)

    return task


def pandas_moving_average_benchmark(group_size=1000, n_groups=1000, window_size=10):
    """Benchmarks Pandas moving average calculation."""
    data = []
    for i in range(n_groups):
        group_data = np.random.rand(group_size)
        group_df = pd.DataFrame({"value": group_data, "group": [i] * group_size})
        data.append(group_df)
    df = pd.concat(data)

    def task():
        df.groupby("group")["value"].rolling(window=window_size).mean().reset_index(
            drop=True
        )

    return task


def pandas_aggregation_benchmark(group_size=1000, n_groups=1000):
    """Benchmarks Pandas aggregation."""
    data = []
    for i in range(n_groups):
        group_data = np.random.rand(group_size)
        group_df = pd.DataFrame({"value": group_data, "group": [i] * group_size})
        data.append(group_df)
    df = pd.concat(data)

    def task():
        df.groupby("group")["value"].sum()

    return task


def lasso_cv_benchmark(n_samples=10000, n_features=100):
    """Benchmarks LassoCV."""
    X = np.random.rand(n_samples, n_features)
    y = np.random.rand(n_samples)

    def task():
        lasso = LassoCV(cv=5, n_jobs=-1, random_state=42)
        lasso.fit(X, y)

    return task


def sgd_classifier_benchmark(n_samples=10000, n_features=100):
    """Benchmarks SGDClassifier."""
    X, y = make_classification(
        n_samples=n_samples, n_features=n_features, random_state=42
    )

    def task():
        sgd = SGDClassifier(loss="hinge", penalty="l2", random_state=42, n_jobs=-1)
        sgd.fit(X, y)

    return task


def lgbm_benchmark(n_samples=10000, n_features=200, n_categorical=20):
    """Benchmarks LightGBM with Optuna hyperparameter tuning."""
    X, y = make_classification(
        n_samples=n_samples, n_features=n_features + n_categorical, random_state=42
    )
    categorical_features = list(range(n_features, n_features + n_categorical))
    X = pd.DataFrame(X)
    for col in categorical_features:
        X[col] = X[col].astype("category")
    X = X.values
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    def objective(trial):
        lgb_params = {
            "objective": "multiclass",
            "num_class": len(np.unique(y)),
            "metric": "multi_logloss",
            "random_state": 42,
            "n_estimators": trial.suggest_int("n_estimators", 100, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "num_leaves": trial.suggest_int("num_leaves", 20, 50),
            "max_depth": trial.suggest_int("max_depth", 5, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 50),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        }
        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        accuracy = sklearn.metrics.accuracy_score(y_test, preds)
        return accuracy

    def task():
        sampler = TPESampler(seed=42)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(
            objective, n_trials=5, n_jobs=1
        )  # n_jobs=1 to avoid nested parallelism

    return task
