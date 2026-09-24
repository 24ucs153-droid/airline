"""
predict.py
----------
Prediction utilities: load a saved pipeline and produce a prediction
(with probability) from a single dict of feature values.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

POSITIVE_LABEL = "satisfied"
NEGATIVE_LABEL = "dissatisfied"


# ---------------------------------------------------------------------------
# Model I/O
# ---------------------------------------------------------------------------


def save_pipeline(pipeline: Any, path: str | Path) -> None:
    """Persist a trained sklearn Pipeline with joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    logger.info("Pipeline saved to: %s", path)


def load_pipeline(path: str | Path) -> Any:
    """Load a previously saved sklearn Pipeline."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found at '{path}'. "
            "Run `python src/train.py` to train and save the model first."
        )
    pipeline = joblib.load(path)
    logger.info("Pipeline loaded from: %s", path)
    return pipeline


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def predict_single(
    pipeline: Any,
    feature_dict: Dict[str, Any],
) -> Tuple[str, float]:
    """
    Produce a prediction for a single observation.

    Parameters
    ----------
    pipeline : trained sklearn Pipeline (preprocessor + RF)
    feature_dict : mapping of feature_name → value

    Returns
    -------
    (label, confidence)
        label : "satisfied" or "dissatisfied"
        confidence : probability of the predicted class (0.0 – 1.0)
    """
    X = pd.DataFrame([feature_dict])

    class_idx: int = int(pipeline.predict(X)[0])
    label = POSITIVE_LABEL if class_idx == 1 else NEGATIVE_LABEL

    if hasattr(pipeline, "predict_proba"):
        proba: np.ndarray = pipeline.predict_proba(X)[0]  # shape (2,)
        confidence = float(proba[class_idx])
        positive_proba = float(proba[1])
    else:
        confidence = 1.0
        positive_proba = float(class_idx)

    return label, confidence, positive_proba


def predict_batch(
    pipeline: Any,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Produce predictions for a DataFrame of observations.

    Returns the input DataFrame enriched with:
      - predicted_label : str
      - confidence      : float
      - satisfied_proba : float
    """
    result = df.copy()
    result["predicted_label"] = [
        POSITIVE_LABEL if p == 1 else NEGATIVE_LABEL
        for p in pipeline.predict(df)
    ]
    if hasattr(pipeline, "predict_proba"):
        probas = pipeline.predict_proba(df)
        result["satisfied_proba"] = probas[:, 1]
        result["confidence"] = probas[
            np.arange(len(result)),
            pipeline.predict(df),
        ]
    return result
