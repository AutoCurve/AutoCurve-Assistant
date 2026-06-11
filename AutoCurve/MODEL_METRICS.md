# AutoCurve — Model Metrics

## Dataset
| Property | Value |
|---|---|
| Source | Craigslist used-car listings (USA) |
| Total records (raw) | 73,082 |
| Records after cleaning | 52,991 |
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
| One-hot encoded categories | 5990 |

## Evaluation
| Property | Value |
|---|---|
| Method | Random train/test split |
| Train split | 80% (42,392 records) |
| Test split | 20% (10,599 records) |
| Random seed | 42 |

## Results
| Metric | Value |
|---|---|
| MAE (Mean Absolute Error) | **$3,077** |
| RMSE (Root Mean Squared Error) | **$5,198** |
| R² Score | **0.8388** |

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
