"""Statistical baseline forecasts and intermittent-demand methods."""
from __future__ import annotations

import numpy as np
import pandas as pd


def baseline_vectorized(df: pd.DataFrame, method: str = "lag1") -> np.ndarray:
    if method == "rolling3":
        return np.clip(df["demand_rolling_mean_3"].fillna(0).values, 0, None)
    return np.clip(df["demand_lag_1"].fillna(0).values, 0, None)


def baseline_for_cluster(df: pd.DataFrame, cluster: str, method: str) -> np.ndarray:
    if cluster == "dead":
        return np.zeros(len(df))
    if cluster == "intermittent":
        lag1 = df["demand_lag_1"].fillna(0).values
        return np.where(lag1 > 0, lag1, 0.0)
    return baseline_vectorized(df, method=method)


def croston_sba(demand: np.ndarray) -> float:
    """SBA variant of Croston on a 1D demand history (non-negative)."""
    d = np.asarray(demand, dtype=float)
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return 0.0
    pos_idx = np.where(d > 0)[0]
    if len(pos_idx) == 0:
        return 0.0
    if len(pos_idx) == 1:
        return float(d[pos_idx[0]])

    sizes = d[pos_idx]
    intervals = np.diff(pos_idx, prepend=pos_idx[0])
    intervals[0] = pos_idx[0] + 1

    z_hat = sizes[0]
    p_hat = intervals[0]
    alpha = 0.1
    for i in range(1, len(pos_idx)):
        z_hat = z_hat + alpha * (sizes[i] - z_hat)
        p_hat = p_hat + alpha * (intervals[i] - p_hat)

    if p_hat <= 0:
        return 0.0
    forecast = (z_hat / p_hat) * (1 - alpha / 2)
    return max(float(forecast), 0.0)


def tsb_forecast(demand: np.ndarray, alpha: float = 0.1, beta: float = 0.1) -> float:
    d = np.asarray(demand, dtype=float)
    if len(d) == 0:
        return 0.0
    p, z = 0.0, 0.0
    for x in d:
        ind = 1.0 if x > 0 else 0.0
        p = p + beta * (ind - p)
        if x > 0:
            z = z + alpha * (x - z) if z > 0 else x
    return max(p * z, 0.0)


def croston_batch(month_df: pd.DataFrame, history: pd.DataFrame) -> np.ndarray:
    """Vectorized Croston per office×SKU using history before forecast month."""
    month = month_df["Date"].iloc[0]
    hist = history[history["Date"] < month]
    if hist.empty:
        return np.zeros(len(month_df))

    cro = hist.groupby(["SalesOfficeCode", "SKU_Code"])["Demand"].apply(
        lambda s: croston_sba(s.values)
    )
    merged = month_df[["SalesOfficeCode", "SKU_Code"]].merge(
        cro.rename("croston").reset_index(), on=["SalesOfficeCode", "SKU_Code"], how="left"
    )
    return np.clip(merged["croston"].fillna(0).values, 0, None)


def intermittent_predict(
    month_df: pd.DataFrame,
    history: pd.DataFrame,
    *,
    use_lag1_cap: bool = True,
) -> np.ndarray:
    cro = croston_batch(month_df, history)
    if not use_lag1_cap:
        return cro
    lag1 = month_df["demand_lag_1"].fillna(0).values
    out = np.where(lag1 > 0, np.minimum(lag1, np.maximum(cro, lag1)), cro)
    return np.clip(out, 0, None)
