"""Volume tier labels for office×SKU series."""
from __future__ import annotations

import pandas as pd

from src.config import FORECAST_START


def assign_volume_tiers(
    df_history: pd.DataFrame,
    *,
    top_pct: float = 0.10,
) -> pd.DataFrame:
    """Label series as top10 / mid / tail by last-12-month demand volume."""
    hist = df_history[df_history["Date"] < FORECAST_START].copy()
    recent = hist[hist["Date"] >= hist["Date"].max() - pd.DateOffset(months=12)]
    vol = (
        recent.groupby(["SalesOfficeCode", "SKU_Code"])["Demand"]
        .sum()
        .reset_index(name="volume")
    )
    vol = vol.sort_values("volume", ascending=False).reset_index(drop=True)
    total_vol = max(float(vol["volume"].sum()), 1.0)
    vol["cum_share"] = vol["volume"].cumsum() / total_vol

    n = len(vol)
    top_n = max(int(n * top_pct), 1)
    vol["volume_tier"] = "tail"
    vol.loc[vol.index < top_n, "volume_tier"] = "top10"
    vol.loc[(vol.index >= top_n) & (vol["cum_share"] <= 0.50), "volume_tier"] = "mid"

    return vol[["SalesOfficeCode", "SKU_Code", "volume_tier", "volume"]]


def attach_volume_tiers(df: pd.DataFrame, tiers: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(tiers[["SalesOfficeCode", "SKU_Code", "volume_tier"]], on=["SalesOfficeCode", "SKU_Code"], how="left")
    return out.assign(volume_tier=lambda d: d["volume_tier"].fillna("tail"))
