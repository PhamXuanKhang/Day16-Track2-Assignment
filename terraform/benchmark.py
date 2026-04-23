#!/usr/bin/env python3
"""
LightGBM benchmark on Credit Card Fraud Detection dataset.
Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
Expected file: ~/ml-benchmark/creditcard.csv (284,807 rows)

Usage:
    python3 benchmark.py
Outputs:
    benchmark_result.json  (same directory)
"""

import json
import time
import os

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

DATASET_PATH = os.path.expanduser("~/ml-benchmark/creditcard.csv")
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_result.json")


def load_data():
    t0 = time.time()
    df = pd.read_csv(DATASET_PATH)
    elapsed = round(time.time() - t0, 4)
    print(f"[data]  Loaded {len(df):,} rows in {elapsed}s")
    X = df.drop(columns=["Class"])
    y = df["Class"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y), elapsed


def train(X_train, y_train):
    train_set = lgb.Dataset(X_train, label=y_train)
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "n_jobs": -1,
        "verbose": -1,
    }
    callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=50)]
    t0 = time.time()
    model = lgb.train(
        params,
        train_set,
        num_boost_round=500,
        valid_sets=[train_set],
        callbacks=callbacks,
    )
    elapsed = round(time.time() - t0, 4)
    print(f"[train] Finished in {elapsed}s — best iteration: {model.best_iteration}")
    return model, elapsed


def evaluate(model, X_test, y_test):
    y_prob = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "auc_roc":   round(roc_auc_score(y_test, y_prob), 6),
        "accuracy":  round(accuracy_score(y_test, y_pred), 6),
        "f1_score":  round(f1_score(y_test, y_pred), 6),
        "precision": round(precision_score(y_test, y_pred), 6),
        "recall":    round(recall_score(y_test, y_pred), 6),
    }


def inference_latency(model, X_test):
    single_row = X_test.iloc[:1]
    batch_1000 = X_test.iloc[:1000]

    # Warm-up
    model.predict(single_row)

    runs = 100
    t0 = time.time()
    for _ in range(runs):
        model.predict(single_row)
    latency_1row_ms = round((time.time() - t0) / runs * 1000, 4)

    t0 = time.time()
    model.predict(batch_1000)
    throughput_1000_ms = round((time.time() - t0) * 1000, 4)

    rows_per_sec = round(1000 / (throughput_1000_ms / 1000))
    print(f"[infer] 1-row latency: {latency_1row_ms}ms | 1000-row batch: {throughput_1000_ms}ms ({rows_per_sec} rows/s)")
    return latency_1row_ms, throughput_1000_ms


def main():
    print("=" * 55)
    print("  LightGBM Benchmark — Credit Card Fraud Detection")
    print("=" * 55)

    (X_train, X_test, y_train, y_test), data_load_time = load_data()
    model, train_time = train(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)
    latency_1row, latency_1000 = inference_latency(model, X_test)

    result = {
        "data_load_time_s":           data_load_time,
        "training_time_s":            train_time,
        "best_iteration":             model.best_iteration,
        "auc_roc":                    metrics["auc_roc"],
        "accuracy":                   metrics["accuracy"],
        "f1_score":                   metrics["f1_score"],
        "precision":                  metrics["precision"],
        "recall":                     metrics["recall"],
        "inference_latency_1row_ms":  latency_1row,
        "inference_latency_1000_ms":  latency_1000,
        "instance_type":              "m7i-flex.large",
        "dataset_rows":               284807,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print("\n--- Results ---")
    for k, v in result.items():
        print(f"  {k:<38} {v}")
    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
