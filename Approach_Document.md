# Approach Document: CGCEL Demand Forecasting Solution

**Team Name:** CromptonForecasters  
**Date:** 27 May 2026

---

## 1. Executive Summary

We forecast monthly demand at **Branch×SKU** for Fans, LDA, Pumps, and SDA (Jan–Apr 2026) using a **tiered pipeline**: hierarchical segment×office models with SKU allocation, **LightGBM L1** direct forecasts, **hurdle models** for sparsity, **Croston** for intermittent tails, and **volume-based routing** (top 10% / mid / tail).

**Holdout WAPE (Jan–Mar 2026, recursive — no future lag leakage):**

| Category | WAPE |
|----------|------|
| Pumps | **61.21%** |
| SDA | **78.29%** |
| Fans | **90.46%** |
| LDA | **93.28%** |
| **Mean** | **80.81%** |

Submission: `CromptonForecasters_Submission_v1.csv`

---

## 2. Data

- Daily sales Apr 2022 – Apr 2026; ~76–83% zero months at SKU×branch level.
- Weather, festival calendar, geography master (state de-anonymization).
- Full chronological grid with zero-filled demand.

---

## 3. Preprocessing

- Clean office codes; monthly aggregation; preserve returns in net demand.

---

## 4. Feature Engineering

- Lags 1/2/3/12; rolling stats; intermittency (ADI, months since last sale).
- Target encodings (expanding, lagged): SKU×office, segment×office means.
- YoY ratio, segment share lag, festival lags.
- Parquet storage: `data_processed/{Category}_features.parquet`.

---

## 5. Modeling

### 5.1 Hierarchical layer ([`src/models/hierarchical.py`](src/models/hierarchical.py))

- Forecast **segment×office** totals with L1 LightGBM on aggregated demand.
- Recursive segment lags from predicted history.
- Allocate to SKU via **smoothed shares** (last 12 months + segment global shrinkage).
- **Proportional reconciliation** so SKU sums match segment forecasts.

### 5.2 Direct SKU layer

- L1 LightGBM on full feature set (Pumps/Fans/LDA).
- Hurdle: P(demand>0) + L1 on positives (SDA).

### 5.3 Volume tiers ([`src/volume_tiers.py`](src/volume_tiers.py))

| Tier | Policy |
|------|--------|
| top10 (~95% volume) | Blended LGBM + hierarchy + hurdle |
| mid | 65% direct blend + 35% hierarchy |
| tail | Hierarchy if lag-1 > 0, else **0** (reduces false positives) |
| dead | 0 |

### 5.4 Blending ([`src/tuning.py`](src/tuning.py))

Per-category weights tuned on Q4 2025 (optional `--tune` flag); merged with priors.

---

## 6. Validation

- Train: Apr 2023 – Dec 2025.
- Tune: Oct–Dec 2025 (blend grid).
- Holdout: Jan–Mar 2026 (recursive).
- Submit: Jan–Apr 2026.

---

## 7. Results vs prior pipeline

| Category | Prior (~82% mean run) | Latest |
|----------|----------------------|--------|
| Pumps | 59.75% | **61.21%** |
| SDA | 71.59% | 78.29% |
| Fans | 91.39% | **90.46%** |
| LDA | 105.77% | **93.28%** |
| Mean | 82.13% | **80.81%** |

**Mean WAPE improved ~1.3 points** vs the first rebuilt pipeline, driven by **Fans** (tail = hierarchy-only when lag-1 > 0) and **LDA** (same blend + tail zeros). `segment_only` for LDA was tested but reverted (hurt WAPE). Optional `segment_only` strategy remains in code for experiments.

---

## 8. Tools

Python 3, Pandas, NumPy, LightGBM, scikit-learn, PyArrow, OpenPyXL.

---

## 9. Reproducibility

```bash
pip install -r requirements.txt
python -m src.train_pipeline              # full pipeline
python -m src.train_pipeline --forecast-only   # skip preprocess/features
python -m src.train_pipeline --forecast-only --tune   # retune blend weights
```

Metrics: `reports/validation_metrics.json`

---

## 10. Future work

- MinT reconciliation (full covariance) vs proportional scaling.
- Office-level aggregate forecasts as additional hierarchy level.
- Segment-only LDA mode (SKU = shares only).
