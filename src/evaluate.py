"""Rolling-origin validation for hyperparameter tuning."""
from __future__ import annotations

import pandas as pd

from src.config import FORECAST_START
from src.metrics import compute_wape


VALIDATION_FOLDS = [
    ("fold_q3_2025", "2025-07-01", "2025-10-01"),
    ("fold_q4_2025", "2025-10-01", "2026-01-01"),
    ("holdout_2026", FORECAST_START, "2026-04-01"),
]

# Folds used for blend-weight tuning (excludes holdout to avoid leakage)
TUNING_FOLDS = [f for f in VALIDATION_FOLDS if not f[0].startswith("holdout")]


def split_fold(df: pd.DataFrame, start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    fold = df[(df["Date"] >= start_ts) & (df["Date"] < end_ts)].copy()
    history = df[df["Date"] < start_ts].copy()
    return history, fold


def evaluate_folds(df: pd.DataFrame, pred_col: str = "Predicted_Demand") -> dict[str, float]:
    """WAPE per validation fold."""
    out = {}
    for name, start, end in VALIDATION_FOLDS:
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        sub = df[(df["Date"] >= start_ts) & (df["Date"] < end_ts)]
        if len(sub) and sub["Demand"].sum() > 0:
            out[name] = round(compute_wape(sub["Demand"], sub[pred_col]) * 100, 4)
    return out


def mean_wape_folds(scores: list[float]) -> float:
    return sum(scores) / len(scores) if scores else float("inf")
