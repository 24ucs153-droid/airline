

"""
data_preprocessing.py
---------------------
Defines the preprocessing pipeline and dataset-loading utilities for the
Airline Customer Satisfaction classifier.

No data leakage: all fit operations happen on training data only, via
scikit-learn Pipeline / ColumnTransformer.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants – discovered programmatically during initial EDA
# ---------------------------------------------------------------------------

TARGET_COLUMN = "satisfaction"
POSITIVE_LABEL = "satisfied"  # mapped → 1
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Categorical feature columns (string dtype)
CATEGORICAL_FEATURES = ["Customer Type", "Type of Travel", "Class"]

# Numerical features = all columns except target and categorical
# The actual list is built dynamically in load_and_split() to stay flexible.

# Rating columns (0-5 Likert scale) – treated as numeric / ordinal
RATING_COLUMNS = [
    "Seat comfort",
    "Departure/Arrival time convenient",
    "Food and drink",
    "Gate location",
    "Inflight wifi service",
    "Inflight entertainment",
    "Online support",
    "Ease of Online booking",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Cleanliness",
    "Online boarding",
]

CONTINUOUS_COLUMNS = [
    "Age",
    "Flight Distance",
    "Departure Delay in Minutes",
    "Arrival Delay in Minutes",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(csv_path: str | Path) -> pd.DataFrame:
    """Load the CSV dataset and return a DataFrame."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    logger.info("Loaded dataset: %d rows × %d columns", *df.shape)
    return df


def validate_dataframe(df: pd.DataFrame) -> None:
    """Run basic sanity checks and log a quality report."""
    # Check target exists
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found. "
            f"Available columns: {df.columns.tolist()}"
        )

    # Missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        logger.warning("Missing values detected:\n%s", missing.to_string())
    else:
        logger.info("No missing values detected.")

    # Duplicates
    n_dup = df.duplicated().sum()
    if n_dup:
        logger.warning("Found %d duplicate rows.", n_dup)
    else:
        logger.info("No duplicate rows.")

    # Target distribution
    dist = df[TARGET_COLUMN].value_counts(dropna=False)
    logger.info("Target distribution:\n%s", dist.to_string())


# ---------------------------------------------------------------------------
# Feature / target split
# ---------------------------------------------------------------------------


def split_features_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) where y is a binary integer series (1 = satisfied)."""
    validate_dataframe(df)

    X = df.drop(columns=[TARGET_COLUMN]).copy()
    y = (df[TARGET_COLUMN] == POSITIVE_LABEL).astype(int)

    logger.info("Features shape: %s  Target classes: %s", X.shape, np.unique(y))
    return X, y


def load_and_split(
    csv_path: str | Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load data, validate, and return (X_train, X_test, y_train, y_test)."""
    df = load_data(csv_path)
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    logger.info(
        "Train size: %d  Test size: %d  Positive-class rate (train): %.2f%%",
        len(X_train),
        len(X_test),
        y_train.mean() * 100,
    )
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Preprocessing pipeline builder
# ---------------------------------------------------------------------------


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
      - Imputes missing values (median for numeric, most-frequent for categorical)
      - Scales continuous numeric features
      - One-hot encodes categorical features
    Rating columns (0-5 integers) are kept as-is after median imputation.
    """
    # Derive numerical columns dynamically (all non-categorical feature cols)
    present_cat = [c for c in CATEGORICAL_FEATURES if c in X.columns]
    all_numeric = [c for c in X.columns if c not in present_cat]
    present_continuous = [c for c in CONTINUOUS_COLUMNS if c in X.columns]
    present_ratings = [c for c in all_numeric if c not in present_continuous]

    logger.info(
        "Preprocessor: continuous=%s | ratings=%s | categorical=%s",
        present_continuous,
        present_ratings,
        present_cat,
    )

    # Pipeline for continuous features: impute → scale
    continuous_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Pipeline for rating features: impute only (already 0-5 integers)
    rating_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    # Pipeline for categorical: impute (most frequent) → OHE
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ohe",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    transformers = [
        ("continuous", continuous_pipe, present_continuous),
        ("ratings", rating_pipe, present_ratings),
        ("categorical", categorical_pipe, present_cat),
    ]

    # Drop any transformer whose feature list is empty
    transformers = [(n, t, c) for (n, t, c) in transformers if c]

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor
