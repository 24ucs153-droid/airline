"""
train.py
--------
Entry-point script for training the Airline Customer Satisfaction model.

Usage
-----
    python src/train.py

This script:
  1. Loads and validates the dataset.
  2. Splits into train / test sets.
  3. Builds the preprocessing + Random Forest pipeline.
  4. (Optional) Runs lightweight RandomizedSearchCV tuning.
  5. Trains the final model.
  6. Evaluates on the held-out test set.
  7. Saves the trained pipeline + evaluation artefacts.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Path setup – allow running from any working directory
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "Airline_customer_satisfaction.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"
METRICS_PATH = PROJECT_ROOT / "models" / "metrics.json"
PLOTS_DIR = PROJECT_ROOT / "models" / "plots"

# Add src to path so sibling imports work
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_preprocessing import build_preprocessor, load_and_split
from evaluate import evaluate_pipeline
from predict import save_pipeline

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hyperparameter search space (kept deliberately narrow for speed)
# ---------------------------------------------------------------------------
PARAM_DIST = {
    "rf__n_estimators": [100, 150, 200],
    "rf__max_depth": [12, 16, 20, None],
    "rf__min_samples_split": [2, 5, 10],
    "rf__min_samples_leaf": [1, 2, 4],
    "rf__max_features": ["sqrt", "log2"],
}

# Set to True to run RandomizedSearchCV (adds ~1-2 min for n_iter=20)
RUN_HYPERPARAMETER_SEARCH = True
N_SEARCH_ITER = 20
CV_FOLDS = 3


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------


def train(run_search: bool = RUN_HYPERPARAMETER_SEARCH) -> None:
    logger.info("=" * 60)
    logger.info("Airline Customer Satisfaction – Random Forest Classifier")
    logger.info("=" * 60)

    # 1. Load data
    logger.info("Loading data from: %s", DATA_PATH)
    X_train, X_test, y_train, y_test = load_and_split(DATA_PATH)

    # 2. Build preprocessor
    preprocessor = build_preprocessor(X_train)

    # 3. Build pipeline with default RF
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("rf", rf),
        ]
    )

    # 4. Optional hyperparameter search
    best_params: dict = {}
    if run_search:
        logger.info(
            "Running RandomizedSearchCV (%d iterations, %d-fold CV) …",
            N_SEARCH_ITER,
            CV_FOLDS,
        )
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=PARAM_DIST,
            n_iter=N_SEARCH_ITER,
            cv=CV_FOLDS,
            scoring="f1",
            n_jobs=-1,
            random_state=42,
            verbose=1,
            refit=True,
        )
        t0 = time.time()
        search.fit(X_train, y_train)
        elapsed = time.time() - t0

        best_params = search.best_params_
        pipeline = search.best_estimator_  # refit=True → already trained

        logger.info("Search done in %.1f s", elapsed)
        logger.info("Best params: %s", best_params)
        logger.info("Best CV F1:  %.4f", search.best_score_)
    else:
        # Train with fixed hyperparameters
        logger.info("Training with fixed hyperparameters …")
        t0 = time.time()
        pipeline.fit(X_train, y_train)
        elapsed = time.time() - t0
        logger.info("Training done in %.1f s", elapsed)

    # 5. Evaluate
    logger.info("Evaluating on held-out test set …")
    metrics = evaluate_pipeline(
        pipeline=pipeline,
        X_test=X_test,
        y_test=y_test,
        plots_dir=PLOTS_DIR,
    )

    # 6. Save pipeline
    save_pipeline(pipeline, MODEL_PATH)

    # 7. Save metrics JSON for the UI to read
    metrics["best_params"] = best_params
    metrics["train_size"] = int(len(X_train))
    metrics["test_size"] = int(len(X_test))
    metrics["feature_columns"] = X_train.columns.tolist()

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to: %s", METRICS_PATH)

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info("  Model : %s", MODEL_PATH)
    logger.info("  Plots : %s", PLOTS_DIR)
    logger.info("=" * 60)


if __name__ == "__main__":
    train()
