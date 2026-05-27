"""Hierarchical forecasting: aggregate LGBM, SKU allocation, reconciliation."""
from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import FORECAST_START, TRAIN_START

AGG_FEATURES = [
    "month_num",
    "quarter",
    "fiscal_month",
    "demand_lag_1",
    "demand_lag_2",
    "demand_lag_3",
    "demand_lag_12",
    "demand_rolling_mean_3",
    "weather_temp_avg",
    "weather_cooling_days",
    "fest_diwali",
    "fest_holi",
    "fest_durga",
]

SEGMENT_KEYS = ["SalesOfficeCode", "Segment"]


def build_aggregate_table(df: pd.DataFrame, group_keys: list[str]) -> pd.DataFrame:
    """Sum demand to monthly aggregate series with lags and calendar features."""
    agg = df.groupby(["Date"] + group_keys, as_index=False)["Demand"].sum()
    agg = agg.sort_values(group_keys + ["Date"]).reset_index(drop=True)

    g = agg.groupby(group_keys, sort=False)
    agg["demand_lag_1"] = g["Demand"].shift(1)
    agg["demand_lag_2"] = g["Demand"].shift(2)
    agg["demand_lag_3"] = g["Demand"].shift(3)
    agg["demand_lag_12"] = g["Demand"].shift(12)
    agg["demand_rolling_mean_3"] = g["demand_lag_1"].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )

    cal = df[["Date", "month_num", "quarter", "fiscal_month", "fest_diwali", "fest_holi", "fest_durga"]].drop_duplicates(
        "Date"
    )
    agg = agg.merge(cal, on="Date", how="left")

    if "SalesOfficeCode" in group_keys:
        weather = (
            df.groupby(["Date", "SalesOfficeCode"])
            .agg(
                weather_temp_avg=("weather_temp_avg", "mean"),
                weather_cooling_days=("weather_cooling_days", "mean"),
            )
            .reset_index()
        )
        agg = agg.merge(weather, on=["Date", "SalesOfficeCode"], how="left")
    else:
        agg["weather_temp_avg"] = 0.0
        agg["weather_cooling_days"] = 0.0

    return agg.fillna(0)


def historical_shares(
    df_hist: pd.DataFrame,
    level_keys: list[str],
    *,
    alpha: float = 0.85,
    share_floor: float = 0.01,
) -> pd.DataFrame:
    """Smoothed SKU shares within segment×office (last 12 months)."""
    recent = df_hist[df_hist["Date"] >= df_hist["Date"].max() - pd.DateOffset(months=12)]
    parent = recent.groupby(level_keys)["Demand"].sum().reset_index(name="parent_total")
    leaf = (
        recent.groupby(["SalesOfficeCode", "SKU_Code", "Segment"])["Demand"]
        .sum()
        .reset_index(name="sku_demand")
    )
    leaf = leaf.merge(parent, on=level_keys, how="left")
    raw = np.where(leaf["parent_total"] > 0, leaf["sku_demand"] / leaf["parent_total"], 0.0)

    sku_seg = recent.groupby(["Segment", "SKU_Code"])["Demand"].sum().reset_index(name="sku_seg_vol")
    seg_tot = recent.groupby("Segment")["Demand"].sum().reset_index(name="seg_vol")
    seg_global = sku_seg.merge(seg_tot, on="Segment", how="left")
    seg_global["global_share"] = np.where(
        seg_global["seg_vol"] > 0, seg_global["sku_seg_vol"] / seg_global["seg_vol"], 0.0
    )
    leaf = leaf.merge(seg_global[["Segment", "SKU_Code", "global_share"]], on=["Segment", "SKU_Code"], how="left")
    leaf["global_share"] = leaf["global_share"].fillna(0)

    leaf["share"] = alpha * raw + (1 - alpha) * leaf["global_share"]
    leaf["share"] = leaf["share"].clip(lower=share_floor)
    # normalize within parent
    tot = leaf.groupby(level_keys)["share"].transform("sum")
    leaf["share"] = np.where(tot > 0, leaf["share"] / tot, 0.0)
    return leaf[["SalesOfficeCode", "SKU_Code", "Segment", "share"]]


def reconcile_proportional(
    sku_preds: np.ndarray,
    df_month: pd.DataFrame,
    segment_targets: pd.DataFrame,
) -> np.ndarray:
    """Scale SKU predictions so sums match segment×office targets per month."""
    out = sku_preds.copy().astype(float)
    tmp = df_month.copy()
    tmp["_pred"] = out
    tmp = tmp.merge(
        segment_targets,
        on=["Date", "SalesOfficeCode", "Segment"],
        how="left",
    )
    tmp["seg_target"] = tmp["seg_pred"].fillna(0)

    for (_, _, _), grp in tmp.groupby(["Date", "SalesOfficeCode", "Segment"]):
        idx = grp.index
        s = grp["_pred"].sum()
        target = grp["seg_target"].iloc[0]
        if s > 0 and target > 0:
            factor = target / s
            out[df_month.index.get_indexer(idx)] = grp["_pred"].values * factor
        elif target > 0 and s <= 0:
            n = len(grp)
            out[df_month.index.get_indexer(idx)] = target / max(n, 1)

    return np.clip(out, 0, None)


@dataclass
class HierarchicalForecaster:
    """Recursive segment×office forecasts allocated to SKU level."""

    df_history: pd.DataFrame
    shares: pd.DataFrame
    model: lgb.LGBMRegressor
    features: list[str]
    seg_series: pd.DataFrame
    seg_preds: dict[tuple, float] = field(default_factory=dict)
    learning_rate: float = 0.08
    _seg_lookup: dict = field(default_factory=dict)

    @classmethod
    def fit(
        cls,
        df_history: pd.DataFrame,
        *,
        share_alpha: float = 0.85,
        learning_rate: float = 0.08,
    ) -> HierarchicalForecaster:
        hist = df_history[df_history["Date"] >= TRAIN_START].copy()
        shares = historical_shares(df_history, SEGMENT_KEYS, alpha=share_alpha)
        seg_series = build_aggregate_table(hist, SEGMENT_KEYS)
        train = seg_series[seg_series["Date"] < FORECAST_START]
        feats = [f for f in AGG_FEATURES if f in train.columns]
        model = lgb.LGBMRegressor(
            objective="regression_l1",
            n_estimators=200,
            learning_rate=learning_rate,
            num_leaves=31,
            random_state=42,
            verbosity=-1,
        )
        model.fit(train[feats], train["Demand"])
        inst = cls(
            df_history=df_history,
            shares=shares,
            model=model,
            features=feats,
            seg_series=seg_series,
            learning_rate=learning_rate,
        )
        inst._seg_lookup = inst.seg_series.set_index(SEGMENT_KEYS + ["Date"])["Demand"].to_dict()
        return inst

    def _get_seg_lag(self, office: str, segment: str, lag_date: pd.Timestamp) -> float:
        if lag_date < pd.Timestamp(FORECAST_START):
            return float(self._seg_lookup.get((office, segment, lag_date), 0.0))
        return float(self.seg_preds.get((office, segment, lag_date), 0.0))

    def _predict_segment_month(self, month: pd.Timestamp, month_df: pd.DataFrame) -> pd.DataFrame:
        """Forecast segment×office totals for combos present in month_df."""
        combos = month_df[SEGMENT_KEYS].drop_duplicates().reset_index(drop=True)
        cal_month = month_df.groupby(SEGMENT_KEYS, as_index=False).first()
        t1, t2, t3, t12 = (
            month - pd.DateOffset(months=1),
            month - pd.DateOffset(months=2),
            month - pd.DateOffset(months=3),
            month - pd.DateOffset(months=12),
        )

        l1 = combos.apply(lambda r: self._get_seg_lag(r["SalesOfficeCode"], r["Segment"], t1), axis=1)
        l2 = combos.apply(lambda r: self._get_seg_lag(r["SalesOfficeCode"], r["Segment"], t2), axis=1)
        l3 = combos.apply(lambda r: self._get_seg_lag(r["SalesOfficeCode"], r["Segment"], t3), axis=1)
        l12 = combos.apply(lambda r: self._get_seg_lag(r["SalesOfficeCode"], r["Segment"], t12), axis=1)

        feat_df = combos.copy()
        feat_df["Date"] = month
        feat_df["demand_lag_1"] = l1.values
        feat_df["demand_lag_2"] = l2.values
        feat_df["demand_lag_3"] = l3.values
        feat_df["demand_lag_12"] = l12.values
        feat_df["demand_rolling_mean_3"] = (feat_df["demand_lag_1"] + feat_df["demand_lag_2"] + feat_df["demand_lag_3"]) / 3
        cal_cols = [c for c in AGG_FEATURES if c in cal_month.columns]
        feat_df = feat_df.merge(cal_month[SEGMENT_KEYS + cal_cols], on=SEGMENT_KEYS, how="left")
        for c in self.features:
            if c not in feat_df.columns:
                feat_df[c] = 0
        preds = np.clip(self.model.predict(feat_df[self.features]), 0, None)
        feat_df["seg_pred"] = preds
        for _, row in feat_df.iterrows():
            key = (row["SalesOfficeCode"], row["Segment"], month)
            self.seg_preds[key] = float(row["seg_pred"])
        return feat_df[["Date", "SalesOfficeCode", "Segment", "seg_pred"]]

    def allocate_to_sku(self, month_df: pd.DataFrame, segment_targets: pd.DataFrame) -> np.ndarray:
        """Top-down: segment forecast × smoothed share."""
        merged = month_df.merge(
            self.shares, on=["SalesOfficeCode", "SKU_Code", "Segment"], how="left"
        )
        merged = merged.merge(
            segment_targets,
            on=["Date", "SalesOfficeCode", "Segment"],
            how="left",
        )
        merged["share"] = merged["share"].fillna(0)
        merged["seg_pred"] = merged["seg_pred"].fillna(0)
        raw = merged["seg_pred"].values * merged["share"].values
        return reconcile_proportional(raw, month_df, segment_targets)

    def predict_month(self, month: pd.Timestamp, month_df: pd.DataFrame) -> np.ndarray:
        seg_targets = self._predict_segment_month(month, month_df)
        return self.allocate_to_sku(month_df, seg_targets)


def forecast_hierarchical_leaf(
    df_history: pd.DataFrame,
    df_target: pd.DataFrame,
    *,
    share_alpha: float = 0.85,
    learning_rate: float = 0.08,
) -> np.ndarray:
    """Full hierarchical forecast for all target rows (recursive by month)."""
    hf = HierarchicalForecaster.fit(
        df_history, share_alpha=share_alpha, learning_rate=learning_rate
    )
    preds = np.zeros(len(df_target))
    for month in sorted(df_target["Date"].unique()):
        mask = df_target["Date"] == month
        month_df = df_target[mask]
        p = hf.predict_month(month, month_df.copy())
        preds[df_target.index.get_indexer(month_df.index)] = p
    return preds
