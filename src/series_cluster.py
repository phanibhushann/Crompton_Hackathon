"""Classify office×SKU series into dead / intermittent / stable."""
from __future__ import annotations

import pandas as pd

from src.config import FORECAST_START


def assign_clusters(df_history: pd.DataFrame) -> pd.DataFrame:
    hist = df_history[df_history["Date"] < FORECAST_START].copy()
    last6_start = pd.Timestamp(FORECAST_START) - pd.DateOffset(months=6)
    hist6 = hist[hist["Date"] >= last6_start]

    stats = hist.groupby(["SalesOfficeCode", "SKU_Code"]).agg(
        nonzero_months=("Demand", lambda x: (x > 0).sum()),
        n_months=("Demand", "count"),
    )
    last6 = hist6.groupby(["SalesOfficeCode", "SKU_Code"])["Demand"].sum().rename("last6_demand")
    stats = stats.join(last6, how="left").fillna(0)
    stats["nonzero_rate"] = stats["nonzero_months"] / stats["n_months"].clip(lower=1)

    if "adi" in hist.columns:
        adi = hist.sort_values("Date").groupby(["SalesOfficeCode", "SKU_Code"])["adi"].last()
        stats = stats.join(adi.rename("adi"), how="left")
    else:
        stats["adi"] = 2.0
    stats["adi"] = stats["adi"].fillna(2.0)

    def label(row) -> str:
        if row["last6_demand"] <= 0:
            return "dead"
        if row["nonzero_rate"] < 0.3 or row["adi"] > 1.32:
            return "intermittent"
        return "stable"

    stats["series_cluster"] = stats.apply(label, axis=1)
    return stats.reset_index()[["SalesOfficeCode", "SKU_Code", "series_cluster"]]


def attach_clusters(df: pd.DataFrame, clusters: pd.DataFrame) -> pd.DataFrame:
    return df.merge(clusters, on=["SalesOfficeCode", "SKU_Code"], how="left").assign(
        series_cluster=lambda d: d["series_cluster"].fillna("intermittent")
    )
