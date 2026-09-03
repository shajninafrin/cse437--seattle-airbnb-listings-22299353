# [Project Title]

**Course:** CSE437, Section [X], [Semester/Year]
**Group members:** [Full Name, Student ID] · [Full Name, Student ID]
**GitHub repository:** [link]
**Date:** [date]

---

## Summary

_150–200 words. State the dataset, the problem, the target variable, the two model families
you compared, your headline metric, and the single most important finding. Write this section
last, after everything else is done._

[Write here.]

---

## 1. Problem and Dataset

### 1.1 Problem statement
We predict the nightly listing price of Seattle Airbnb listings from host, property, location,
and review/availability features. Accurate, interpretable price prediction matters to
[hosts pricing new listings / the platform's pricing-guidance tools / researchers studying
short-term rental markets — pick and expand on the framing that fits your angle].

[Expand to 1–2 paragraphs.]

### 1.2 Dataset
**Source:** Seattle Airbnb Listings, Kaggle (`shanelev/seattle-airbnb-listings`),
https://www.kaggle.com/datasets/shanelev/seattle-airbnb-listings — an Inside Airbnb-format
scrape of Seattle listings.
**Collection method:** [state what the Kaggle page says about how the data was collected/scraped]
**Rows / columns:** [fill in from `df_raw.shape` in notebook 01]
**Time period covered:** [fill in from the Kaggle page / scrape date in the data]
**Licence / terms of use:** [state what the Kaggle page says]

### 1.3 Target variable
`log_price` — the log-transformed (`log1p`), outlier-capped nightly price. Continuous variable.
Distribution shown in Figure [X] (`figures/02_log_price_distribution.png`); raw price is
right-skewed (skew = [fill in from notebook 02 data journal]), reduced substantially after the
log transform (skew = [fill in]).

### 1.4 Three questions
1. Which features (location, room type, number of reviews, availability) most strongly affect
   Airbnb price in Seattle?
2. Can we accurately predict listing price using host and property features?
3. Which listings does the model predict poorly, and why?

---

## 2. Data Handling and Preprocessing

### 2.1 Data quality audit
[Summarize from notebook 01: missing values per column (top offenders), duplicate rows found,
inconsistent categories noticed, impossible values found (e.g. $0 prices).]

### 2.2 Missing values
[State the mechanism assumed per column type and the strategy: columns >70% missing dropped
(list them and why); review scores treated as structurally missing when `number_of_reviews == 0`
and imputed; property-size fields imputed with room-type-grouped median; host rate fields
percentage-parsed then median-imputed. Justify each choice in 1–2 sentences.]

### 2.3 Outliers
Detection method: 1st/99th percentile capping on price (`notebook 02`).
[Fill in: n_rows_capped, price_cap_bounds from the notebook 02 data journal output.]

### 2.4 Transformation and scaling
- Price parsed from `"$1,234.00"` string format to float.
- Target log-transformed (`log1p`) after capping.
- Amenities and host_verifications (list-valued string columns) converted to a small set of
  boolean flags for common values plus a total count, rather than full one-hot encoding.
- Numeric features scaled with `StandardScaler`, categorical features one-hot encoded — both
  fit **only on the training split**, inside a `ColumnTransformer` + `Pipeline`
  (`notebooks/04_modeling_and_tuning.ipynb`), to prevent leakage.

**Leakage guard, stated explicitly:** all imputers/scalers/encoders are fit inside the
`sklearn.Pipeline` on `X_train` only; `X_val`/`X_test` are only ever `.transform()`-ed, never
`.fit()`-ed on.

### 2.5 Before and after

| Stage | Rows | Columns |
|---|---|---|
| Raw | [fill in] | [fill in] |
| After price-validity filter | [fill in] | [fill in] |
| After cleaning (final) | [fill in] | [fill in] |

[Pull these numbers from the `summary_table` printed in notebook 02.]

---

## 3. Statistical Analysis

### 3.1 Descriptive statistics
[Select 4-6 numeric features and summarize central tendency/spread/shape from
`df[desc_cols].describe().T` in notebook 02/03 — don't paste the raw table, comment on what's
notable, e.g. "accommodates ranges from 1–X with a median of Y..." Include a frequency table
for room_type and/or neighbourhood.]

### 3.2 Relationships
[Include 2-3 figures: the correlation heatmap (`figures/`), price-by-room-type boxplot, price
vs. number-of-reviews scatter — from notebook 01/02. State what each shows in one sentence.]

### 3.3 What the data says so far
- [Observation 1]
- [Observation 2]
- [Observation 3]
- [Observation 4, optional]
- [Observation 5, optional]

---

## 4. Feature Engineering

### 4.1 Derived features
- `n_amenities`, `n_verifications` — total counts.
- Amenity flags (`has_wifi`, `has_kitchen`, etc.) for the ~13 most common amenities.
- Verification flags (`verified_email`, `verified_phone`, etc.) for the 4 most common
  verification types.
- `has_reviews` — flags structurally-missing review scores.
- `neighbourhood_grouped`, `property_type_grouped` — rare categorical levels collapsed into
  `'Other'` to control dimensionality.

Reasoning: raw amenities/host_verifications are list-valued strings with hundreds of unique
values; full one-hot encoding would be mostly noise. Flags for common, plausibly price-relevant
values keep the signal while controlling dimensionality (faculty guidance).

### 4.2 Dimensionality reduction
PCA applied to the numeric/flag feature block (`notebook 03`). [Fill in: components needed for
90% variance, out of how many total features.] **Not included in the final modeling pipeline** —
most features carry distinct, non-redundant signal, and keeping raw features preserves
interpretability needed for RQ1/RQ3. Dimensionality was instead controlled via variance/
correlation filtering and categorical-level grouping (see 4.3).

### 4.3 Feature selection
Method: variance threshold (drop near-constant flags) → correlation-with-target filter (drop
|corr| < 0.02) → pairwise collinearity filter (drop one of any pair with |corr| > 0.9, keeping
whichever correlates more with the target). [Fill in from notebook 03 output: which features
were dropped at each stage.]

### 4.4 Final feature set
**Numeric/flag features ([N]):** [paste from notebook 03 `final_features.txt`]
**Categorical features ([N]):** [paste from notebook 03]

[Write the justification paragraph: what's in, what's out, why. E.g. "Location is represented
via neighbourhood_grouped rather than raw latitude/longitude, avoiding a redundant location
signal. Amenities are represented via flags for the most common items plus a count rather than
the full vocabulary..."]

---

## 5. Modeling and Validation

### 5.1 Validation strategy
70/15/15 train/validation/test split (`train_test_split`, `random_state=42`). 5-fold
cross-validation used inside hyperparameter search on the training split. No temporal or group
structure requiring special handling — listings are treated as i.i.d.

### 5.2 Baseline
`DummyRegressor(strategy='median')` — predicts the median `log_price` for every listing.
Test RMSE (log space): [fill in from notebook 05 `results_df`]. Test RMSE ($): [fill in].

### 5.3 Model families
1. **Ridge Regression** — linear, interpretable via coefficients, assumes an additive linear
   relationship between standardized features and log price; regularization (L2) controls
   overfitting on correlated/one-hot features.
2. **Random Forest** — non-linear ensemble of decision trees, captures interactions and
   non-linear effects without requiring feature scaling; less interpretable per-coefficient but
   supports feature importance and permutation importance.

### 5.4 Metrics
Primary metric: **RMSE** (in dollars, after `expm1` back-transform) — directly interpretable as
"average dollar error," and appropriate for a continuous regression target. Reported alongside
MAE (robust to the influence of any remaining large residuals) and R² (log space, for
variance-explained context). Stated as primary before results were seen, consistent with the
modeling notebook.

---

## 6. Hyperparameter Tuning

### 6.1 Search space

| Model | Hyperparameter | Range/Grid |
|---|---|---|
| Ridge | `alpha` | 20 values, log-spaced 1e-3 to 1e3 |
| Random Forest | `n_estimators` | [100, 200, 400, 600] |
| Random Forest | `max_depth` | [None, 5, 10, 20, 30] |
| Random Forest | `min_samples_split` | [2, 5, 10] |
| Random Forest | `min_samples_leaf` | [1, 2, 4] |
| Random Forest | `max_features` | ['sqrt', 'log2', 0.5] |

### 6.2 Method
Ridge: `GridSearchCV`, full grid (20 candidates), 5-fold CV, scoring =
`neg_root_mean_squared_error` (100 fits total).
Random Forest: `RandomizedSearchCV`, 30 sampled candidates out of 540 possible combinations,
5-fold CV, same scoring (150 fits total) — random search used given the larger combinatorial
space, for efficiency.

### 6.3 Results

| Model | Best config | Best CV RMSE (log space) |
|---|---|---|
| Ridge | [fill in `best_params_`] | [fill in] |
| Random Forest | [fill in `best_params_`] | [fill in] |

[Include the Ridge alpha-vs-CV-score trend figure from `figures/04_ridge_alpha_search.png` and
describe the trend in 1-2 sentences — e.g. "CV RMSE improves as alpha increases from 1e-3 then
worsens past [X], indicating mild regularization is helpful but heavy regularization
underfits."]

---

## 7. Results, Visualization and Error Analysis

### 7.1 Test set performance

| Model | RMSE (log) | MAE (log) | R² (log) | RMSE ($) | MAE ($) |
|---|---|---|---|---|---|
| Baseline (median) | [fill in] | [fill in] | [fill in] | [fill in] | [fill in] |
| Ridge (tuned) | [fill in] | [fill in] | [fill in] | [fill in] | [fill in] |
| Random Forest (tuned) | [fill in] | [fill in] | [fill in] | [fill in] | [fill in] |

[Pull directly from `data/processed/test_results.csv` / notebook 05.]

### 7.2 Visualization
- Feature importance / coefficients: `figures/05_ridge_coefficients.png`,
  `figures/05_rf_importance.png`, `figures/05_permutation_importance.png`
- Predicted vs. actual and residuals: `figures/05_predicted_vs_actual_and_residuals.png`

[Embed the images in the PDF version and describe what each shows in 1-2 sentences.]

### 7.3 Error analysis
[Write the discussion from notebook 05's error-analysis cells: which room type/neighbourhood
has the worst mean absolute error, whether residuals show heteroscedasticity, whether
low-review-count listings are harder to predict. Include **at least two concrete examples**
from `data/processed/worst_predictions.csv` with their actual vs. predicted price and your
explanation for why each is hard.]

### 7.4 Answers to your three questions

**RQ1 — Which features most strongly affect price?**
[Answer directly, citing the coefficient/importance plots.]

**RQ2 — Can we accurately predict listing price using host and property features?**
[Answer directly, citing the test RMSE/MAE/R² and improvement over baseline.]

**RQ3 — Which listings does the model predict poorly, and why?**
[Answer directly, citing the error analysis findings.]

---

## 8. Limitations and Next Steps

[Honest constraints — e.g.: single-city, single-snapshot scrape; listed price may not equal
actual booked rate; amenity flag list is a subjective subset, not the full vocabulary;
small-sample neighbourhoods have noisier group statistics; no seasonality data if this is a
single-point-in-time scrape. What you'd do with more time/data: e.g. gradient boosting,
richer location features, temporal pricing data.]

---

## 9. Contributions

| Member | Student ID | Contribution |
|---|---|---|
| [Name] | [ID] | [What they actually did] |
| [Name] | [ID] | [What they actually did] |

---

## References

- Dataset: Seattle Airbnb Listings, Kaggle, `shanelev/seattle-airbnb-listings`,
  https://www.kaggle.com/datasets/shanelev/seattle-airbnb-listings
- Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, joblib
- **AI assistance:** Claude (Anthropic) was used to scaffold notebook structure/boilerplate
  code, the report template, and repo layout per the assignment spec. All data analysis
  decisions, model results, and written interpretation are the authors' own.
