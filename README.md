# Airline Customer Satisfaction Predictor

A complete end-to-end Machine Learning project that predicts whether an airline
passenger is **satisfied** or **dissatisfied** with their flight experience.

Trained on **129 880** real passenger records using a **Random Forest** classifier
wrapped in a full scikit-learn preprocessing pipeline.

---

## Dataset Description

| Property | Value |
|---|---|
| Source | Airline Passenger Satisfaction (Kaggle) |
| File | `data/Airline_customer_satisfaction.csv` |
| Rows | 129 880 |
| Columns | 22 |
| Problem type | **Binary Classification** |
| Target column | `satisfaction` (`satisfied` / `dissatisfied`) |
| Class balance | 54.7% satisfied / 45.3% dissatisfied |

### Features

| Feature | Type | Description |
|---|---|---|
| Customer Type | Categorical | Loyal / Disloyal customer |
| Age | Numerical | Passenger age (7–85) |
| Type of Travel | Categorical | Business / Personal |
| Class | Categorical | Business / Eco Plus / Eco |
| Flight Distance | Numerical | Distance in miles |
| Departure Delay in Minutes | Numerical | Departure delay |
| Arrival Delay in Minutes | Numerical | Arrival delay (had 393 NaN → median-imputed) |
| Seat comfort (0–5) | Rating | Likert-scale service rating |
| Inflight wifi service (0–5) | Rating | |
| Inflight entertainment (0–5) | Rating | |
| Food and drink (0–5) | Rating | |
| Gate location (0–5) | Rating | |
| Departure/Arrival time convenient (0–5) | Rating | |
| Online support (0–5) | Rating | |
| Ease of Online booking (0–5) | Rating | |
| On-board service (0–5) | Rating | |
| Leg room service (0–5) | Rating | |
| Baggage handling (0–5) | Rating | |
| Checkin service (0–5) | Rating | |
| Cleanliness (0–5) | Rating | |
| Online boarding (0–5) | Rating | |

---

## Preprocessing Steps

1. **Missing values** – `Arrival Delay in Minutes` (393 NaN) → median imputation
2. **Continuous features** (`Age`, `Flight Distance`, delays) → median imputation + StandardScaler
3. **Rating features** (0–5 integers) → median imputation; kept as integers
4. **Categorical features** → most-frequent imputation + OneHotEncoder (`handle_unknown='ignore'`)
5. **Train/test split** – 80/20, stratified on the target, `random_state=42`
6. **No data leakage** – all fit operations use scikit-learn Pipeline + ColumnTransformer

---

## Why Random Forest?

* Handles mixed feature types (numerical, ordinal ratings, one-hot categoricals) without normalisation requirements
* Robust to outliers (e.g., very large delay values)
* Provides built-in feature importance rankings
* Excellent baseline performance with minimal tuning
* Supports probability calibration for confidence scores

---

## Model Evaluation Results

| Metric | Value |
|---|---|
| Accuracy | **~95.5%** |
| Precision | ~96.8% |
| Recall | ~94.9% |
| F1 Score | ~95.9% |
| ROC-AUC | **~0.993** |

> Exact results are saved to `models/metrics.json` after training.

---

## Project Structure

```text
airline/
├── data/
│   └── Airline_customer_satisfaction.csv   ← dataset
├── models/
│   ├── model.joblib                         ← trained pipeline (created after training)
│   ├── metrics.json                         ← evaluation metrics JSON
│   └── plots/
│       ├── confusion_matrix.png
│       ├── roc_curve.png
│       └── feature_importance.png
├── src/
│   ├── data_preprocessing.py               ← data loading & preprocessing pipeline
│   ├── train.py                             ← training script (main entry-point)
│   ├── evaluate.py                          ← evaluation & visualisation helpers
│   └── predict.py                           ← inference utilities
├── app/
│   └── app.py                               ← Streamlit UI
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

### Requirements
- Python 3.9+
- Anaconda / pip

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## How to Train the Model

```bash
python src/train.py
```

This will:
1. Load and validate the dataset
2. Split data 80/20 (stratified)
3. Run `RandomizedSearchCV` (20 iterations, 3-fold CV) to find optimal hyperparameters
4. Train the final pipeline on the full training set
5. Evaluate on the held-out test set
6. Save the model to `models/model.joblib`
7. Save metrics to `models/metrics.json`
8. Save plots to `models/plots/`

Training takes approximately **2–5 minutes** depending on your hardware.

---

## How to Start the UI

```bash
streamlit run app/app.py
```

The app will open at `http://localhost:8501`.

---

## Example Usage

### CLI prediction

```python
from src.predict import load_pipeline, predict_single

pipeline = load_pipeline("models/model.joblib")

sample = {
    "Customer Type": "Loyal Customer",
    "Age": 35,
    "Type of Travel": "Business travel",
    "Class": "Business",
    "Flight Distance": 2000,
    "Seat comfort": 4,
    "Departure/Arrival time convenient": 3,
    "Food and drink": 4,
    "Gate location": 3,
    "Inflight wifi service": 4,
    "Inflight entertainment": 5,
    "Online support": 4,
    "Ease of Online booking": 4,
    "On-board service": 4,
    "Leg room service": 4,
    "Baggage handling": 4,
    "Checkin service": 4,
    "Cleanliness": 4,
    "Online boarding": 4,
    "Departure Delay in Minutes": 0,
    "Arrival Delay in Minutes": 0.0,
}

label, confidence, pos_proba = predict_single(pipeline, sample)
print(f"Prediction: {label}  (confidence: {confidence*100:.1f}%)")
```

---

## Known Limitations

* The model was trained on one specific airline satisfaction dataset; generalisation to other airlines or geographies is not guaranteed.
* Passengers who did not rate a service (value = 0) are treated as missing/neutral; this may introduce noise.
* The high class balance (54.7/45.3) means accuracy is a reasonable metric, but F1 and ROC-AUC are more informative.
* Delay values above 1584 minutes were not seen during training; predictions for extreme outliers may be less reliable.
