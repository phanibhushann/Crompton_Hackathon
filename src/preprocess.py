"""Load raw Excel sales data and aggregate to monthly branch-SKU demand."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CATEGORY_FILES, PROCESSED_DIR, RAW_DATA_DIR, STATE_MAPPING


def clean_state(state) -> str | float:
    if pd.isnull(state):
        return np.nan
    return STATE_MAPPING.get(str(state).strip(), str(state).strip())


def _geo_sheet_name(xl: pd.ExcelFile) -> str:
    for name in xl.sheet_names:
        if "Geography" in name or "Geao" in name:
            return name
    raise ValueError(f"No geography sheet in {xl.sheet_names}")


def preprocess_category(category: str) -> pd.DataFrame:
    filename, sheets = CATEGORY_FILES[category]
    path = RAW_DATA_DIR / filename
    print(f"\n--- Preprocessing {category} from {filename} ---")

    dfs = [pd.read_excel(path, sheet_name=s) for s in sheets]
    df_sales = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df_sales):,} daily rows")

    df_sales["SalesOfficeCode"] = df_sales["SalesOfficeCode"].astype(str).str.strip()
    df_sales["SalesOfficeCode"] = df_sales["SalesOfficeCode"].replace(
        {"Tot FouTd": np.nan, "Not Found": np.nan, "#T/A": np.nan, "nan": np.nan}
    )
    before = len(df_sales)
    df_sales = df_sales.dropna(subset=["SalesOfficeCode"])
    print(f"Dropped {before - len(df_sales):,} rows with invalid office codes")

    xl = pd.ExcelFile(path)
    df_geo = pd.read_excel(path, sheet_name=_geo_sheet_name(xl))
    df_geo = df_geo.dropna(subset=["SalesOfficeCode"])
    df_geo["SalesOfficeCode"] = df_geo["SalesOfficeCode"].astype(str).str.strip()
    df_geo["State"] = df_geo["State"].apply(clean_state)

    for col in ("State", "Region"):
        if col in df_sales.columns:
            df_sales = df_sales.drop(columns=[col])

    df_sales = df_sales.merge(
        df_geo[["SalesOfficeCode", "Region", "State"]], on="SalesOfficeCode", how="left"
    )
    df_sales = df_sales.rename(columns={"State": "SalesOffice_Name"})

    sku_col = "Alternative Sku Code"
    df_sales[sku_col] = df_sales[sku_col].astype(str).str.strip()
    df_sales["Segment"] = df_sales["Segment"].astype(str).str.strip()
    df_sales = df_sales.dropna(subset=[sku_col])

    df_sales["InvoiceDate_dt"] = pd.to_datetime(df_sales["InvoiceDate"], errors="coerce")
    df_sales = df_sales.dropna(subset=["InvoiceDate_dt"])
    df_sales["Year"] = df_sales["InvoiceDate_dt"].dt.year
    df_sales["Month_Name"] = df_sales["InvoiceDate_dt"].dt.strftime("%B")

    df_monthly = (
        df_sales.groupby(
            ["Year", "Month_Name", "Region", "SalesOfficeCode", "SalesOffice_Name", "Segment", sku_col],
            as_index=False,
        )["Total Quantity"]
        .sum()
        .rename(columns={sku_col: "SKU_Code", "Total Quantity": "Demand"})
    )

    print(f"Monthly shape: {df_monthly.shape}")
    print(f"Unique SKUs: {df_monthly['SKU_Code'].nunique():,}")
    print(f"Unique office×SKU: {df_monthly.groupby(['SalesOfficeCode', 'SKU_Code']).ngroups:,}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / f"{category}_monthly_sales.csv"
    df_monthly.to_csv(out, index=False)
    print(f"Saved {out}")
    return df_monthly


def run_all(categories: list[str] | None = None) -> None:
    from src.config import CATEGORIES

    for cat in categories or CATEGORIES:
        preprocess_category(cat)
    print("\n=== Preprocessing complete ===")


if __name__ == "__main__":
    run_all()
