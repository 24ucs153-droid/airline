"""
evaluate.py
-----------
Evaluation utilities for the Airline Customer Satisfaction classifier.
Computes metrics and generates visualisation plots saved to disk.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> Dict[str, float]:
    """Return a dict of classification metrics."""
    metrics: Dict[str, float] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)

    for k, v in metrics.items():
        logger.info("  %s: %.4f", k, v)

    return metrics


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------


def _save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: %s", path)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Tuple[str, str] = ("Dissatisfied", "Satisfied"),
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot and optionally save a confusion matrix heat-map."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_xlabel("Predicted", fontsize=13)
    ax.set_ylabel("Actual", fontsize=13)
    ax.set_title("Confusion Matrix", fontsize=15, fontweight="bold")
    fig.tight_layout()

    if save_path:
        _save_fig(fig, save_path)

    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot and optionally save the ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {auc:.4f})", color="#4C72B0")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title("ROC Curve", fontsize=15, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path:
        _save_fig(fig, save_path)

    return fig


def plot_feature_importance(
    pipeline: Any,
    feature_names: list[str],
    top_n: int = 20,
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Extract feature importances from the Random Forest inside a sklearn
    Pipeline and plot a horizontal bar chart.
    """
    rf = pipeline.named_steps["rf"]
    importances = rf.feature_importances_

    # Map importances back to original feature names via the ColumnTransformer
    preprocessor = pipeline.named_steps["preprocessor"]
    try:
        ohe_feature_names = preprocessor.get_feature_names_out()
    except AttributeError:
        ohe_feature_names = [f"feat_{i}" for i in range(len(importances))]

    importance_df = (
        pd.DataFrame({"feature": ohe_feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )

    # Clean up OHE prefixes for readability
    importance_df["feature"] = (
        importance_df["feature"]
        .str.replace(r"^(continuous__|ratings__|categorical__)", "", regex=True)
        .str.replace("categorical__", "", regex=False)
    )

    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.35)))
    bars = ax.barh(
        importance_df["feature"][::-1],
        importance_df["importance"][::-1],
        color=sns.color_palette("Blues_r", n_colors=top_n),
    )
    ax.set_xlabel("Importance", fontsize=13)
    ax.set_title(f"Top-{top_n} Feature Importances", fontsize=15, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    if save_path:
        _save_fig(fig, save_path)

    return fig


# ---------------------------------------------------------------------------
# Full evaluation run
# ---------------------------------------------------------------------------


def evaluate_pipeline(
    pipeline: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    plots_dir: Path | None = None,
) -> Dict[str, float]:
    """
    Run full evaluation: compute metrics and generate all plots.

    Returns
    -------
    Dict with accuracy, precision, recall, f1, roc_auc.
    """
    logger.info("Running evaluation …")

    y_pred = pipeline.predict(X_test)
    y_proba: np.ndarray | None = None
    if hasattr(pipeline, "predict_proba"):
        y_proba = pipeline.predict_proba(X_test)[:, 1]

    # Metrics
    metrics = compute_metrics(y_test.values, y_pred, y_proba)

    # Print report
    print("\n" + "=" * 50)
    print("CLASSIFICATION REPORT")
    print("=" * 50)
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Dissatisfied", "Satisfied"],
            digits=4,
        )
    )

    for k, v in metrics.items():
        print(f"  {k:12s}: {v:.4f}")

    # Plots
    if plots_dir is not None:
        plots_dir = Path(plots_dir)
        plot_confusion_matrix(
            y_test.values, y_pred, save_path=plots_dir / "confusion_matrix.png"
        )
        if y_proba is not None:
            plot_roc_curve(
                y_test.values, y_proba, save_path=plots_dir / "roc_curve.png"
            )
        plot_feature_importance(
            pipeline,
            feature_names=list(X_test.columns),
            save_path=plots_dir / "feature_importance.png",
        )

    return metrics
