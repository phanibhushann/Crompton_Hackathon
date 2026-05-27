"""Build full timeline grid, lags, weather, festivals, intermittency, encodings."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CATEGORIES, HISTORY_START, PROCESSED_DIR, RAW_DATA_DIR


def process_weather() -> pd.DataFrame:
    path = RAW_DATA_DIR / "weather_data.xlsx"
    df = pd.read_excel(path, sheet_name="Data")
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df["Year"] = df["date_dt"].dt.year
    df["Month_Name"] = df["date_dt"].dt.strftime("%B")

    for col in (
        "temp_avg_value",
        "temp_max_value",
        "temp_min_value",
        "cooling_value",
        "heating_value",
        "precipitation_value",
    ):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["hot_day"] = (df["temp_max_value"] > 35).astype(int)
    df["cold_day"] = (df["temp_min_value"] < 15).astype(int)

    return (
        df.groupby(["SalesOfficeCode", "Year", "Month_Name"])
        .agg(
            weather_temp_avg=("temp_avg_value", "mean"),
            weather_temp_max=("temp_max_value", "mean"),
            weather_temp_min=("temp_min_value", "mean"),
            weather_cooling_days=("cooling_value", "sum"),
            weather_heating_days=("heating_value", "sum"),
            weather_precip=("precipitation_value", "sum"),
            weather_hot_days_count=("hot_day", "sum"),
            weather_cold_days_count=("cold_day", "sum"),
        )
        .reset_index()
    )


def process_festivals() -> pd.DataFrame:
    path = RAW_DATA_DIR / "Fans_Data.xlsx"
    df = pd.read_excel(path, sheet_name="Festival Calendar")
    df["Date_dt"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Year"] = df["Date_dt"].dt.year
    df["Month_Name"] = df["Date_dt"].dt.strftime("%B")
    df["fest_diwali"] = df["Festival"].str.contains("Diwali", case=False, na=False).astype(int)
    df["fest_holi"] = df["Festival"].str.contains("Holi", case=False, na=False).astype(int)
    df["fest_eid"] = df["Festival"].str.contains("Eid", case=False, na=False).astype(int)
    df["fest_durga"] = df["Festival"].str.contains("Durga|Dussehra", case=False, na=False).astype(int)
    return (
        df.groupby(["Year", "Month_Name"])
        .agg(
            fest_count=("Festival", "count"),
            fest_diwali=("fest_diwali", "max"),
            fest_holi=("fest_holi", "max"),
            fest_eid=("fest_eid", "max"),
            fest_durga=("fest_durga", "max"),
        )
        .reset_index()
    )


def _add_intermittency(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["SalesOfficeCode", "SKU_Code"], sort=False)

    def months_since_last(x: pd.Series) -> pd.Series:
        out, since = [], 999
        for v in x:
            since = 0 if v > 0 else min(since + 1, 999)
            out.append(since)
        return pd.Series(out, index=x.index)

    df["months_since_last_sale"] = g["Demand"].transform(months_since_last)
    df["nonzero_count_6m"] = g["Demand"].transform(lambda x: (x > 0).rolling(6, min_periods=1).sum())
    df["nonzero_count_12m"] = g["Demand"].transform(lambda x: (x > 0).rolling(12, min_periods=1).sum())
    df["adi"] = 12.0 / df["nonzero_count_12m"].clip(lower=1)
    df["cv2"] = g["Demand"].transform(
        lambda x: x.rolling(12, min_periods=2).apply(
            lambda w: (w[w > 0].std() / w[w > 0].mean()) ** 2
            if (w > 0).sum() >= 2 and w[w > 0].mean() > 0
            else 0.0,
            raw=True,
        )
    ).fillna(0)
    return df


def _add_target_encoding(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["SalesOfficeCode", "SKU_Code", "Date"]).reset_index(drop=True)

    def expanding_mean_lagged(series: pd.Series) -> pd.Series:
        return series.expanding().mean().shift(1)

    df["sku_office_mean"] = (
        df.groupby(["SalesOfficeCode", "SKU_Code"])["Demand"]
        .transform(expanding_mean_lagged)
        .fillna(0)
    )
    df["sku_global_mean"] = df.groupby("SKU_Code")["Demand"].transform(expanding_mean_lagged).fillna(0)
    df["segment_office_mean"] = (
        df.groupby(["SalesOfficeCode", "Segment"])["Demand"]
        .transform(expanding_mean_lagged)
        .fillna(0)
    )
    months_seen = df.groupby(["SalesOfficeCode", "SKU_Code"]).cumcount() + 1
    nonzero_cum = df.groupby(["SalesOfficeCode", "SKU_Code"])["Demand"].transform(
        lambda x: (x > 0).cumsum()
    )
    df["sku_freq"] = (nonzero_cum / months_seen).fillna(0)
    df["sku_bucket"] = pd.util.hash_pandas_object(df["SKU_Code"], index=False).astype(np.int64) % 256
    df["sku_bucket"] = df["sku_bucket"].astype(str)
    return df


def build_features(category: str, df_weather: pd.DataFrame, df_fest: pd.DataFrame) -> pd.DataFrame:
    print(f"\n=== Feature engineering: {category} ===")
    df = pd.read_csv(PROCESSED_DIR / f"{category}_monthly_sales.csv")

    dates = pd.date_range(start=HISTORY_START, end="2026-04-01", freq="MS")
    df_dates = pd.DataFrame({"Date": dates})
    df_dates["Year"] = df_dates["Date"].dt.year
    df_dates["Month_Name"] = df_dates["Date"].dt.strftime("%B")

    meta = df[["SalesOfficeCode", "SKU_Code", "Region", "SalesOffice_Name", "Segment"]].drop_duplicates()
    meta = meta.dropna(subset=["SKU_Code"])
    meta["Segment"] = meta.groupby("SKU_Code")["Segment"].transform(lambda x: x.ffill().bfill())
    meta["Segment"] = meta["Segment"].fillna("UNKNOWN")
    meta = meta.drop_duplicates()

    grid = df_dates.assign(_k=1).merge(meta.assign(_k=1), on="_k").drop(columns="_k")

    df["Month_Name"] = df["Month_Name"].astype(str).str.strip()
    grid["Month_Name"] = grid["Month_Name"].astype(str).str.strip()
    grid["SKU_Code"] = grid["SKU_Code"].astype(str).str.strip()
    grid["SalesOfficeCode"] = grid["SalesOfficeCode"].astype(str).str.strip()

    grid = grid.merge(
        df[["Year", "Month_Name", "SalesOfficeCode", "SKU_Code", "Demand"]],
        on=["Year", "Month_Name", "SalesOfficeCode", "SKU_Code"],
        how="left",
    )
    grid["Demand"] = grid["Demand"].fillna(0)
    grid = grid.sort_values(["SalesOfficeCode", "SKU_Code", "Date"]).reset_index(drop=True)

    gcols = ["SalesOfficeCode", "SKU_Code"]
    grid["demand_lag_1"] = grid.groupby(gcols)["Demand"].shift(1)
    grid["demand_lag_2"] = grid.groupby(gcols)["Demand"].shift(2)
    grid["demand_lag_3"] = grid.groupby(gcols)["Demand"].shift(3)
    grid["demand_lag_12"] = grid.groupby(gcols)["Demand"].shift(12)
    grid["demand_rolling_mean_3"] = grid.groupby(gcols)["demand_lag_1"].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    grid["demand_rolling_mean_6"] = grid.groupby(gcols)["demand_lag_1"].transform(
        lambda x: x.rolling(6, min_periods=1).mean()
    )
    grid["demand_rolling_std_3"] = (
        grid.groupby(gcols)["demand_lag_1"].transform(lambda x: x.rolling(3, min_periods=1).std()).fillna(0)
    )

    grid["demand_yoy_ratio"] = grid["demand_lag_1"] / (grid["demand_lag_12"] + 1.0)
    _lag1 = grid.groupby(gcols)["Demand"].shift(1)
    seg_sum = grid.assign(_lag1=_lag1).groupby(["Date", "SalesOfficeCode", "Segment"])["_lag1"].transform("sum")
    grid["segment_share_lag"] = (_lag1 / (seg_sum + 1.0)).fillna(0)
    grid = _add_intermittency(grid)
    grid = _add_target_encoding(grid)

    grid = grid.merge(df_weather, on=["SalesOfficeCode", "Year", "Month_Name"], how="left")
    weather_cols = [c for c in grid.columns if c.startswith("weather_")]
    for col in weather_cols:
        clim = grid.groupby(["SalesOfficeCode", "Month_Name"])[col].transform("mean")
        grid[col] = grid[col].fillna(clim).fillna(0)

    grid = grid.merge(df_fest, on=["Year", "Month_Name"], how="left")
    for col in [c for c in grid.columns if c.startswith("fest_")]:
        grid[col] = grid[col].fillna(0)

    grid["fest_diwali_lag1"] = grid.groupby(gcols)["fest_diwali"].shift(1).fillna(0)
    grid["fest_diwali_lag2"] = grid.groupby(gcols)["fest_diwali"].shift(2).fillna(0)

    grid["month_num"] = grid["Date"].dt.month
    grid["quarter"] = grid["Date"].dt.quarter
    grid["fiscal_month"] = grid["month_num"].apply(lambda m: m - 3 if m >= 4 else m + 9)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = PROCESSED_DIR / f"{category}_features.parquet"
    grid.to_parquet(parquet_path, index=False)
    print(f"Saved {parquet_path} ({len(grid):,} rows)")
    return grid


def run_all(categories: list[str] | None = None) -> None:
    df_w = process_weather()
    df_f = process_festivals()
    for cat in categories or CATEGORIES:
        build_features(cat, df_w, df_f)
    print("\n=== Feature engineering complete ===")


if __name__ == "__main__":
    run_all()
