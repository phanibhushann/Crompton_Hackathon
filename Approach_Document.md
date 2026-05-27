# Approach Document: CGCEL Demand Forecasting Solution

**Team Name:** Shackle Not Found  
**Team Members:** PhaniBhushan, Aryan Shrivastava  
**Date:** 27 May 2026  
**Submission File:** `Shackle_Not_Found_Submission_v1.csv`

---

## 1. Executive Summary

We forecast monthly demand at **Branch × SKU** granularity for four categories (Fans, LDA, Pumps, SDA) for **Jan–Apr 2026**.  
The final system is a **tiered hybrid forecasting pipeline** combining:

- Hierarchical segment×office forecasting with SKU allocation
- Direct SKU-level LightGBM models
- Hurdle modeling for sparse demand
- Intermittent-demand baselines (Croston family)
- Volume-tier routing (top10 / mid / tail / dead)

This architecture was selected because SKU×branch demand is highly sparse and unstable, while aggregate segment×office signals are more learnable and robust.

**Final validation (Jan–Mar 2026 holdout, recursive / leakage-safe):**

| Category | WAPE (%) | RMSE |
|---|---:|---:|
| Pumps | **58.83** | **241.82** |
| SDA | **76.75** | **137.62** |
| Fans | **90.16** | **598.25** |
| LDA | **95.08** | **74.07** |
| **Mean WAPE** | **80.20** | n/a |

---

## 2. Data Understanding & EDA

### 2.1 Data sources

- Daily transaction data (Apr 2022–Apr 2026) for Fans, LDA, Pumps, SDA
- Weather data by sales office
- Festival calendar (Diwali, Holi, Eid, Durga/Dussehra)
- Geography master with hashed-state mapping

### 2.2 Key observations

- Demand at branch×SKU monthly level is **highly sparse** (\(\sim 76\%-83\%\) zero months).
- Few high-volume series drive most business demand; long tail dominates row count.
- Strong seasonality exists in categories like Fans and Pumps.
- Festival effects are localized and lagged rather than purely same-month.
- Returns/negative transactions are present and retained in net demand.

### 2.3 Data quality and missingness

- Weather fields include missing values handled through grouped aggregation/fill strategy.
- Office/state identifiers include anonymized/hashes requiring controlled mapping.
- Sparse series frequently contain short non-zero bursts separated by long zero runs.

---

## 3. Data Preprocessing & Cleaning

Key preprocessing steps:

1. Parse and normalize date fields across all category sheets.
2. Standardize office codes and geography keys.
3. Map hashed state identifiers to readable state names.
4. Aggregate daily transactions into monthly branch×SKU demand.
5. Build a full chronological grid and fill missing branch×SKU months with zero demand.
6. Preserve negative values/returns in monthly net demand.

Processed monthly and feature outputs are saved in `data_processed/`.

---

## 4. Feature Engineering

### 4.1 Seasonal / cyclical features

- `month_num`, `quarter`, `fiscal_month`
- Year-over-year ratio proxies (`demand_yoy_ratio`)

### 4.2 Weather features

- Average/max/min temperature
- Cooling/heating degree style aggregates
- Precipitation
- Hot-day and cold-day counts

### 4.3 Festival / holiday indicators

- Festival counts
- Binary festival flags (`fest_diwali`, `fest_holi`, `fest_eid`, `fest_durga`)
- Lagged festival effects (`fest_diwali_lag1`, `fest_diwali_lag2`)

### 4.4 Lag and rolling features

- Lags: 1, 2, 3, 12 months
- Rolling: mean(3), mean(6), std(3)
- Recursive lag updates in inference to avoid future leakage

### 4.5 Intermittency and demand behavior

- ADI (`adi`)
- `months_since_last_sale`
- Non-zero counts in rolling windows
- \(CV^2\) (`cv2`) on non-zero demand

### 4.6 Encodings

- Expanding lagged target encodings:
  - `sku_office_mean`
  - `sku_global_mean`
  - `segment_office_mean`
- Frequency and bucket features (`sku_freq`, `sku_bucket`)

---

## 5. Model Selection & Experimentation

### 5.1 Models tried

- **Direct LightGBM L1** at SKU×branch
- **Hierarchical LightGBM** at segment×office + allocation to SKU
- **Hurdle model** (classification for demand>0 + L1 regression for positives)
- **Baselines**: lag-based and Croston-style intermittent forecasts
- **Blended ensembles** across components

### 5.2 Why this model family

- LightGBM with L1 objective aligns better with WAPE than L2 for sparse data.
- Hierarchical layer improves stability where SKU-level signal is weak.
- Hurdle setup is effective for zero-inflated demand (especially SDA).
- Blending reduces model-specific failure modes across categories.

### 5.3 Hyperparameter / tuning approach

- Category-specific priors for blend weights and thresholds
- Grid search over blend weight combinations
- Prior-regularized merge of tuned weights
- Threshold tuning for hurdle classifier cutoff
- Multi-fold blend tuning on Q3 and Q4 2025 (holdout excluded)

### 5.4 Final model strategy by category

- **Pumps:** LGBM-primary blend
- **SDA:** Hurdle-heavy blend
- **Fans/LDA:** Hierarchy + direct blend with conservative tail handling

---

## 6. Validation Strategy

Validation follows strict time-series logic with no future leakage.

- Training history up to pre-forecast boundary
- Fold Q3 2025: Jul–Sep 2025
- Fold Q4 2025: Oct–Dec 2025
- Holdout 2026: Jan–Mar 2026
- Submission horizon: Jan–Apr 2026

Forecasting is **recursive multi-step**: predicted values from earlier months feed later lag features, matching production behavior.

---

## 7. Results & Model Performance

### 7.1 Final accuracy

- **Primary metric:** WAPE
- **Mean holdout WAPE:** **80.20%**
- **RMSE (Jan–Mar 2026 holdout)** is also reported (see Table in Section 1)

### 7.2 Interpretation

- Pumps is the strongest category and responds well to direct LGBM.
- SDA benefits from hurdle architecture but remains sparse.
- Fans/LDA remain challenging due to high intermittency and long-tail effects.
- Tail-tier WAPE is high by row count, but top-volume tier drives most business impact.

### 7.3 Baseline comparison

Simple lag/intermittent baselines were used as lower-bound references and as blend components.  
Hybrid blended models consistently outperformed single baseline strategies.

### 7.4 RMSE and SHAP note

- WAPE is used as the primary optimization target during tuning.
- RMSE is computed on the same recursive Jan–Mar 2026 holdout (no additional leakage).
- SHAP-based interpretability is not currently automated in the pipeline; it can be added if requested.

---

## 8. Tools & Technologies Used

### 8.1 Programming languages

- Python

### 8.2 ML/statistical libraries

- LightGBM
- scikit-learn
- Custom Croston-style intermittent baseline
- statsmodels (installed for optional statistical baselines)

### 8.3 Data handling

- Pandas
- NumPy
- OpenPyXL (Excel I/O)
- PyArrow (Parquet)

### 8.4 Visualization

- (Primarily notebook/table-based analysis in this submission)

### 8.5 Development environment

- VS Code / Cursor IDE

---

## 9. Key Insights & Recommendations for CGCEL

### 9.1 Where error concentrates (top10 vs tail)

Because the dataset is dominated by intermittent/tail series, error magnitude differs strongly by volume tier. The table below summarizes WAPE (%) on the same Jan–Mar 2026 holdout.

| Category | Top10 WAPE (%) | Tail WAPE (%) |
|---|---:|---:|
| Pumps | 56.93 | 101.61 |
| SDA | 70.64 | 105.38 |
| Fans | 74.07 | 100.62 |
| LDA | 88.65 | 106.71 |

Planning takeaway: **top-volume series are substantially more reliable** than tail series, so safety stock and replenishment policies should treat tail SKUs more conservatively (and leverage hierarchy when last-month sales are zero).

1. **Prioritize top-volume branch×SKU series** for planning decisions.  
   These series contribute most of realized demand and are modeled with richer ensembles.

2. **Use conservative stocking for tail SKUs** with long zero runs.  
   Tail-series false positives are costly and add noise to planning.

3. **Use weather-sensitive planning windows** for seasonal categories (especially Fans).  
   Weather features materially influence category-level demand timing.

4. **Plan around festival lead/lag effects**, not only event month totals.  
   Lagged festival features indicate demand shifts across adjacent months.

5. **Deploy category-specific replenishment policy**, not one global rule.  
   Pumps, SDA, Fans, and LDA have different sparsity/seasonality regimes.

---

## 10. Limitations & Future Improvements

### Current limitations

- Very high sparsity at SKU×branch level limits deterministic accuracy.
- Tail-tier forecasts remain difficult; WAPE can exceed 100% in sparse segments.
- Current reconciliation is proportional, not covariance-optimal.
- SHAP reporting is not currently automated in the pipeline (RMSE is automated and reported).

### Future improvements

- MinT-style reconciliation experiments
- Office-level intermediate hierarchy
- Expanded multi-fold / Bayesian tuning
- Optional SHAP-based interpretability pack for judges
- Additional external regressors (if permitted and available)

---

## 11. References

1. Ke, G. et al. (2017), *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*.  
2. Hyndman, R. J., & Athanasopoulos, G. (2021), *Forecasting: Principles and Practice*.  
3. Croston, J. D. (1972), *Forecasting and stock control for intermittent demands*.  
4. CGCEL Demand Forecasting Competition Brief (official problem statement).
