"""Recursive multi-step forecasting with full lag feature updates."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import BASE_FEATURES, CAT_FEATURES, FORECAST_START


def _history_lookup(df_history: pd.DataFrame) -> dict:
    return df_history.set_index(["SalesOfficeCode", "SKU_Code", "Date"])["Demand"].to_dict()


def _get_lag(
    target_date: pd.Timestamp,
    keys: list[tuple],
    history_lookup: dict,
    df_target: pd.DataFrame,
) -> np.ndarray:
    if target_date < pd.Timestamp(FORECAST_START):
        return np.array([history_lookup.get((c, s, target_date), 0.0) for c, s in keys])
    sub = df_target[df_target["Date"] == target_date]
    lk = dict(zip(zip(sub["SalesOfficeCode"], sub["SKU_Code"]), sub["Predicted_Demand"]))
    return np.array([lk.get((c, s), 0.0) for c, s in keys])


def update_lag_features(df_target: pd.DataFrame, month: pd.Timestamp, history_lookup: dict) -> None:
    mask = df_target["Date"] == month
    keys = list(
        zip(
            df_target.loc[mask, "SalesOfficeCode"],
            df_target.loc[mask, "SKU_Code"],
        )
    )
    t1 = month - pd.DateOffset(months=1)
    t2 = month - pd.DateOffset(months=2)
    t3 = month - pd.DateOffset(months=3)
    t4 = month - pd.DateOffset(months=4)
    t5 = month - pd.DateOffset(months=5)
    t6 = month - pd.DateOffset(months=6)
    t12 = month - pd.DateOffset(months=12)

    l1 = _get_lag(t1, keys, history_lookup, df_target)
    l2 = _get_lag(t2, keys, history_lookup, df_target)
    l3 = _get_lag(t3, keys, history_lookup, df_target)
    l4 = _get_lag(t4, keys, history_lookup, df_target)
    l5 = _get_lag(t5, keys, history_lookup, df_target)
    l6 = _get_lag(t6, keys, history_lookup, df_target)
    l12 = _get_lag(t12, keys, history_lookup, df_target)

    df_target.loc[mask, "demand_lag_1"] = l1
    df_target.loc[mask, "demand_lag_2"] = l2
    df_target.loc[mask, "demand_lag_3"] = l3
    df_target.loc[mask, "demand_lag_12"] = l12

    stacked_3 = np.stack([l1, l2, l3], axis=1)
    df_target.loc[mask, "demand_rolling_mean_3"] = stacked_3.mean(axis=1)
    df_target.loc[mask, "demand_rolling_std_3"] = stacked_3.std(axis=1)
    stacked_6 = np.stack([l1, l2, l3, l4, l5, l6], axis=1)
    df_target.loc[mask, "demand_rolling_mean_6"] = stacked_6.mean(axis=1)


def recursive_predict(
    df_history: pd.DataFrame,
    df_target: pd.DataFrame,
    predict_fn,
    features: list[str],
) -> pd.DataFrame:
    """
    predict_fn: callable(df_month_rows) -> np.ndarray of predictions
    """
    df_target = df_target.copy()
    df_target["Predicted_Demand"] = np.nan
    history_lookup = _history_lookup(df_history)

    for month in sorted(df_target["Date"].unique()):
        update_lag_features(df_target, month, history_lookup)
        mask = df_target["Date"] == month
        month_df = df_target[mask]
        preds = predict_fn(month_df)
        df_target.loc[mask, "Predicted_Demand"] = np.clip(preds, 0, None)

    return df_target
