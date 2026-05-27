"""WAPE and evaluation helpers."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def compute_wape(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    total = actual.sum()
    if total <= 0:
        return 0.0
    return float(np.sum(np.abs(actual - predicted)) / total)


def evaluate_frame(df: pd.DataFrame, actual_col: str = "Demand", pred_col: str = "Predicted_Demand") -> dict:
    mask = df[actual_col].notna()
    wape = compute_wape(df.loc[mask, actual_col].values, df.loc[mask, pred_col].values)
    return {"wape": wape, "wape_pct": wape * 100, "rows": int(mask.sum())}


def save_metrics(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
