"""
Model evaluation script for AutoCurve.
Trains a Linear Regression on the cleaned dataset and reports MAE, RMSE, and R².

Usage:
    python evaluate_model.py
"""
import os
import math
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import backend

FEATURES = ["year", "odometer", "manufacturer", "model", "condition", "fuel", "transmission", "drive"]
TARGET = "price"
TEST_SIZE = 0.20
RANDOM_STATE = 42

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "database.xlsx")


def evaluate():
    print("Loading and cleaning data…")
    df = backend.load_data(DATA_PATH)
    if df is None or df.empty:
        raise RuntimeError("Could not load database.xlsx")

    # Drop rows missing any required feature or target
    cols_needed = FEATURES + [TARGET]
    df = df.dropna(subset=cols_needed)
    df = df[cols_needed].copy()

    total_records = len(df)
    print(f"Records after cleaning: {total_records:,}")

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    numeric_features     = ["year", "odometer"]
    categorical_features = ["manufacturer", "model", "condition", "fuel", "transmission", "drive"]

    preprocessor = ColumnTransformer(transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
    ])

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor",    LinearRegression()),
    ])

    print("Training Linear Regression…")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    print("\n── Evaluation Results ──────────────────")
    print(f"  Records total  : {total_records:,}")
    print(f"  Train size     : {len(X_train):,}  ({100*(1-TEST_SIZE):.0f}%)")
    print(f"  Test size      : {len(X_test):,}   ({100*TEST_SIZE:.0f}%)")
    print(f"  MAE            : ${mae:,.0f}")
    print(f"  RMSE           : ${rmse:,.0f}")
    print(f"  R²             : {r2:.4f}")
    print("────────────────────────────────────────\n")

    # Write MODEL_METRICS.md
    n_categories = sum(
        len(pipeline.named_steps["preprocessor"]
            .transformers_[1][1]
            .categories_[i])
        for i in range(len(categorical_features))
    )

    md = f"""# AutoCurve — Model Metrics

## Dataset
| Property | Value |
|---|---|
| Source | Craigslist used-car listings (USA) |
| Total records (raw) | 73,082 |
| Records after cleaning | {total_records:,} |
| Price filter | $500 – $200,000 |
| Odometer filter | 0 – 1,000,000 miles |
| Year filter | 1980 – 2026 |

## Model
| Property | Value |
|---|---|
| Algorithm | Linear Regression (`sklearn.linear_model.LinearRegression`) |
| Framework | scikit-learn |
| Preprocessing | PassThrough for numerics; OneHotEncoder for categoricals |

## Features
| Type | Features |
|---|---|
| Numeric | `year`, `odometer` |
| Categorical | `manufacturer`, `model`, `condition`, `fuel`, `transmission`, `drive` |
| One-hot encoded categories | {n_categories} |

## Evaluation
| Property | Value |
|---|---|
| Method | Random train/test split |
| Train split | {100*(1-TEST_SIZE):.0f}% ({len(X_train):,} records) |
| Test split | {100*TEST_SIZE:.0f}% ({len(X_test):,} records) |
| Random seed | {RANDOM_STATE} |

## Results
| Metric | Value |
|---|---|
| MAE (Mean Absolute Error) | **${mae:,.0f}** |
| RMSE (Root Mean Squared Error) | **${rmse:,.0f}** |
| R² Score | **{r2:.4f}** |

## Notes
- The **in-app valuation** uses a different approach than the global model evaluated here.
  The app runs per-make/model regressions on year and odometer, combined with a
  nearest-neighbour categorical match — a weighted ensemble tailored to each query.
  The global Linear Regression above is evaluated for resume/portfolio purposes.
- The R² score reflects the inherent variability in used-car prices from subjective
  listing data, which limits ceiling performance for any regression approach.

## Limitations
- Training data is USA Craigslist listings only — international markets not covered.
- Luxury/exotic vehicles are under-represented.
- Listing prices ≠ final sale prices (asking price bias).
- No location/seasonality features.
- Model condition ratings are self-reported by sellers.

## Future Improvements
- Add VIN-based feature lookup (trim, options, accident history).
- Include ZIP code / regional market pricing.
- Try gradient-boosted trees (XGBoost/LightGBM) for non-linear relationships.
- Ensemble the global regression with the per-make/model regression.
- Train on verified sale prices instead of listing prices.
"""

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MODEL_METRICS.md")
    with open(out_path, "w") as f:
        f.write(md)
    print(f"Metrics written to: {out_path}")

    return {"mae": mae, "rmse": rmse, "r2": r2, "total_records": total_records}


if __name__ == "__main__":
    evaluate()
