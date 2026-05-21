"""
traditional_models.py
---------------------
Logistic Regression and Naive Bayes pipelines with TF-IDF vectorization.
Includes GridSearchCV tuning and evaluation.
"""

import time
import logging
import argparse
import json
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from preprocess import get_default_preprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------

def load_imdb(max_train: int = 25000, max_test: int = 5000) -> Tuple:
    """Load and preprocess the IMDb dataset."""
    logger.info("Loading IMDb dataset …")
    dataset = load_dataset("imdb")

    preprocessor = get_default_preprocessor()

    def preprocess_split(split, max_samples):
        texts = dataset[split]["text"][:max_samples]
        labels = dataset[split]["label"][:max_samples]
        logger.info(f"Preprocessing {split} split ({len(texts)} samples) …")
        cleaned = preprocessor.process_batch(texts, verbose=True)
        return cleaned, labels

    X_train, y_train = preprocess_split("train", max_train)
    X_test, y_test = preprocess_split("test", max_test)

    return X_train, y_train, X_test, y_test


# -----------------------------------------------------------------------
# Pipeline builders
# -----------------------------------------------------------------------

def build_logistic_regression_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=50_000,
            sublinear_tf=True,
            min_df=2,
        )),
        ("clf", LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            random_state=42,
        )),
    ])


def build_naive_bayes_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=50_000,
            sublinear_tf=False,   # NB requires non-negative inputs
            min_df=2,
        )),
        ("clf", MultinomialNB(alpha=0.1)),
    ])


# -----------------------------------------------------------------------
# Hyperparameter grids
# -----------------------------------------------------------------------

LR_PARAM_GRID = {
    "tfidf__max_features": [30_000, 50_000],
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "clf__C": [0.1, 1.0, 10.0],
}

NB_PARAM_GRID = {
    "tfidf__max_features": [30_000, 50_000],
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "clf__alpha": [0.01, 0.1, 1.0],
}


# -----------------------------------------------------------------------
# Training & evaluation
# -----------------------------------------------------------------------

def train_and_evaluate(
    name: str,
    pipeline: Pipeline,
    param_grid: Dict[str, Any],
    X_train, y_train, X_test, y_test,
    cv_folds: int = 5,
    tune: bool = True,
) -> Dict[str, Any]:
    """Train a pipeline (with optional GridSearchCV) and return metrics."""

    logger.info(f"\n{'='*60}")
    logger.info(f"  Training: {name}")
    logger.info(f"{'='*60}")

    t0 = time.time()

    if tune:
        logger.info("Running GridSearchCV …")
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        searcher = GridSearchCV(
            pipeline, param_grid, cv=cv,
            scoring="f1_macro", n_jobs=-1, verbose=1,
        )
        searcher.fit(X_train, y_train)
        best_model = searcher.best_estimator_
        logger.info(f"Best params: {searcher.best_params_}")
    else:
        best_model = pipeline.fit(X_train, y_train)

    train_time = time.time() - t0

    # --- Inference ---
    t1 = time.time()
    y_pred = best_model.predict(X_test)
    infer_time = time.time() - t1

    # --- Metrics ---
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")
    cm = confusion_matrix(y_test, y_pred).tolist()

    results = {
        "model": name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "train_time_sec": round(train_time, 2),
        "infer_time_sec": round(infer_time, 4),
        "infer_ms_per_sample": round((infer_time / len(X_test)) * 1000, 4),
        "confusion_matrix": cm,
    }

    logger.info(f"\n{classification_report(y_test, y_pred, target_names=['Negative', 'Positive'])}")
    logger.info(f"Accuracy      : {acc:.4f}")
    logger.info(f"F1 (weighted) : {f1:.4f}")
    logger.info(f"Train time    : {train_time:.2f}s")
    logger.info(f"Infer latency : {results['infer_ms_per_sample']:.4f} ms/sample")

    return results


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main(args):
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_test, y_test = load_imdb()

    all_results = []

    # Naive Bayes
    nb_results = train_and_evaluate(
        name="Naive Bayes (TF-IDF)",
        pipeline=build_naive_bayes_pipeline(),
        param_grid=NB_PARAM_GRID,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        tune=args.tune,
    )
    all_results.append(nb_results)

    # Logistic Regression
    lr_results = train_and_evaluate(
        name="Logistic Regression (TF-IDF)",
        pipeline=build_logistic_regression_pipeline(),
        param_grid=LR_PARAM_GRID,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        tune=args.tune,
    )
    all_results.append(lr_results)

    # Save results
    out_path = output_dir / "traditional_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\n✅ Results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark traditional NLP models")
    parser.add_argument("--output", default="../results", help="Output directory")
    parser.add_argument("--tune", action="store_true", help="Run GridSearchCV tuning")
    args = parser.parse_args()
    main(args)
