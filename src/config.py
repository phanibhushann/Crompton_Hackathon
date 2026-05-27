"""Pipeline configuration — paths relative to repo root."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = REPO_ROOT / "Sales Data"
PROCESSED_DIR = REPO_ROOT / "data_processed"
ARTIFACTS_DIR = REPO_ROOT / "models" / "artifacts"
SUBMISSIONS_DIR = REPO_ROOT / "submissions"
REPORTS_DIR = REPO_ROOT / "reports"

TEAM_NAME = "CromptonForecasters"
SUBMISSION_FILE = REPO_ROOT / f"{TEAM_NAME}_Submission_v1.csv"

CATEGORIES = ["Fans", "LDA", "Pumps", "SDA"]

TRAIN_START = "2023-04-01"
HISTORY_START = "2022-04-01"
FORECAST_START = "2026-01-01"
FORECAST_END = "2026-04-01"
VALIDATION_END = "2026-04-01"

STATE_MAPPING = {
    "5DC36933AAC0190B": "Delhi (NCT)",
    "F54463B32C20E526": "Uttar Pradesh",
    "FFDC55D1D730F9A0": "Haryana",
    "9B5FB605B25C68ED": "Punjab",
    "562BB6FF82BEA5FA": "Rajasthan",
    "87009AD1E0AE2BB5": "West Bengal",
    "403FB14808A17D28": "Assam",
    "257ED1B3C2EDBA8F": "Jharkhand",
    "6C3B8FA244228A94": "Bihar",
    "9878A77B95DD41C1": "Odisha",
    "B206DD96787255B2": "Maharashtra",
    "950CC761E6105A23": "Gujarat",
    "ED3CC80C0D87DB70": "Madhya Pradesh",
    "20083C0767929E44": "Chhattisgarh",
    "6504FD33404CBB66": "Tamil Nadu",
    "734390698195C66F": "Karnataka",
    "4E879D0D3A817714": "Telangana",
    "6F362208E2441C57": "Andhra Pradesh",
    "F105582383EA3198": "Kerala",
}

CATEGORY_FILES = {
    "Fans": ("Fans_Data.xlsx", ["SalesData_22to25", "SalesData_25to26"]),
    "LDA": ("LDA_Data.xlsx", ["SalesData"]),
    "Pumps": ("Pumps_Data.xlsx", ["Sales Data"]),
    "SDA": ("SDA_Data.xlsx", ["Sales Data"]),
}

BASE_FEATURES = [
    "month_num",
    "quarter",
    "fiscal_month",
    "demand_lag_1",
    "demand_lag_2",
    "demand_lag_3",
    "demand_lag_12",
    "demand_rolling_mean_3",
    "demand_rolling_mean_6",
    "demand_rolling_std_3",
    "demand_yoy_ratio",
    "segment_share_lag",
    "fest_diwali_lag1",
    "fest_diwali_lag2",
    "months_since_last_sale",
    "nonzero_count_6m",
    "nonzero_count_12m",
    "adi",
    "cv2",
    "sku_office_mean",
    "sku_global_mean",
    "segment_office_mean",
    "sku_freq",
    "sku_bucket",
    "weather_temp_avg",
    "weather_temp_max",
    "weather_temp_min",
    "weather_cooling_days",
    "weather_heating_days",
    "weather_precip",
    "weather_hot_days_count",
    "weather_cold_days_count",
    "fest_count",
    "fest_diwali",
    "fest_holi",
    "fest_eid",
    "fest_durga",
    "Region",
    "SalesOfficeCode",
    "Segment",
]

CAT_FEATURES = ["Region", "SalesOfficeCode", "Segment", "sku_bucket"]
