"""End-to-end training, hierarchical forecasting, tuning, and submission."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import (
    BASE_FEATURES,
    CAT_FEATURES,
    CATEGORIES,
    FORECAST_END,
    FORECAST_START,
    PROCESSED_DIR,
    REPORTS_DIR,
    SUBMISSION_FILE,
    SUBMISSIONS_DIR,
    TEAM_NAME,
    TRAIN_START,
)
from src.feature_engineering import run_all as run_features
from src.metrics import compute_wape, save_metrics
from src.models.baselines import (
    baseline_for_cluster,
    baseline_vectorized,
    intermittent_predict,
)
from src.models.blend import blend_predictions
from src.models.hierarchical import HierarchicalForecaster
from src.models.hurdle import train_hurdle, tune_threshold
from src.preprocess import run_all as run_preprocess
from src.recursive_forecast import recursive_predict
from src.series_cluster import assign_clusters, attach_clusters
from src.evaluate import TUNING_FOLDS, evaluate_folds
from src.tuning import (
    blend_dict_from_priors,
    default_category_params,
    load_tuning,
    save_tuning,
    tune_blend_multi_fold,
)
from src.volume_tiers import assign_volume_tiers, attach_volume_tiers

CATEGORY_LR = {"Fans": 0.05, "LDA": 0.05, "Pumps": 0.1, "SDA": 0.1}
BASELINE_METHOD = {"Fans": "lag1", "LDA": "lag1", "Pumps": "rolling3", "SDA": "lag1"}
MID_TIER_HIER_WEIGHT = 0.45
MID_TIER_DIRECT_WEIGHT = 0.55
TUNE_SUBSAMPLE_SIZE = 80_000


def load_features(category: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{category}_features.parquet"
    df = pd.read_parquet(path)
    df["Date"] = pd.to_datetime(df["Date"])
    for col in ("Region", "Segment", "SalesOfficeCode", "sku_bucket"):
        if col in df.columns:
            df[col] = df[col].fillna("UNKNOWN").astype(str)
    return df


def _build_blend_tune_fold(
    df: pd.DataFrame,
    fold_start: str,
    fold_end: str,
    *,
    features: list[str],
    share_alpha: float,
    lr: float,
    bl_method: str,
    hurdle,
    lgbm_direct,
) -> tuple[dict[str, np.ndarray], np.ndarray] | None:
    """Build blend components for one tuning fold (history ends before fold_start)."""
    start_ts, end_ts = pd.Timestamp(fold_start), pd.Timestamp(fold_end)
    tune_fold = df[(df["Date"] >= start_ts) & (df["Date"] < end_ts)].copy()
    if tune_fold.empty or tune_fold["Demand"].sum() <= 0:
        return None

    hist = df[df["Date"] < start_ts].copy()
    if len(tune_fold) > TUNE_SUBSAMPLE_SIZE:
        tune_fold = tune_fold.sample(TUNE_SUBSAMPLE_SIZE, random_state=42)
    X_tune = _prep_X(tune_fold, features)

    hf_tune = HierarchicalForecaster.fit(hist, share_alpha=share_alpha, learning_rate=lr)
    hier_tune = np.zeros(len(tune_fold))
    for month in sorted(tune_fold["Date"].unique()):
        mmask = tune_fold["Date"] == month
        hier_tune[mmask.to_numpy()] = hf_tune.predict_month(month, tune_fold[mmask])

    components: dict[str, np.ndarray] = {
        "baseline": baseline_vectorized(tune_fold, bl_method),
        "hierarchical": hier_tune,
    }
    if hurdle is not None:
        components["hurdle"] = hurdle.predict(X_tune)
    if lgbm_direct is not None:
        components["lgbm"] = np.clip(lgbm_direct.predict(X_tune), 0, None)
    return components, tune_fold["Demand"].values.astype(float)


def _prep_X(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    X = df[features].copy()
    for col in CAT_FEATURES:
        if col in X.columns:
            X[col] = X[col].astype("category")
    return X


def _detailed_metrics(val: pd.DataFrame, tiers: pd.DataFrame | None = None) -> dict:
    out = {"overall_wape_pct": round(compute_wape(val["Demand"], val["Predicted_Demand"]) * 100, 4)}
    for m in sorted(val["Date"].unique()):
        sub = val[val["Date"] == m]
        out[f"wape_{m.strftime('%Y_%m')}"] = round(
            compute_wape(sub["Demand"], sub["Predicted_Demand"]) * 100, 4
        )
    if tiers is not None:
        merged = val.merge(
            tiers[["SalesOfficeCode", "SKU_Code", "volume_tier"]],
            on=["SalesOfficeCode", "SKU_Code"],
            how="left",
        )
        vt_col = "volume_tier" if "volume_tier" in merged.columns else "volume_tier_y"
        for tier in ("top10", "mid", "tail"):
            sub = merged[merged[vt_col] == tier]
            if len(sub):
                out[f"wape_tier_{tier}"] = round(
                    compute_wape(sub["Demand"], sub["Predicted_Demand"]) * 100, 4
                )
    return out


def forecast_category(
    category: str,
    *,
    tune: bool = True,
    params: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    print(f"\n{'=' * 20} {category} {'=' * 20}")
    df = load_features(category)
    clusters = assign_clusters(df[df["Date"] < FORECAST_START])
    tiers = assign_volume_tiers(df[df["Date"] < FORECAST_START])
    df = attach_clusters(df, clusters)
    df = attach_volume_tiers(df, tiers)

    df_history = df[df["Date"] < FORECAST_START].copy()
    df_train = df[(df["Date"] >= TRAIN_START) & (df["Date"] < FORECAST_START)].copy()
    df_target = df[(df["Date"] >= FORECAST_START) & (df["Date"] <= FORECAST_END)].copy()

    features = [f for f in BASE_FEATURES if f in df.columns]
    lr = CATEGORY_LR[category]
    bl_method = BASELINE_METHOD[category]

    cat_params = params if params is not None else default_category_params(category)
    if tune and params is None:
        cached = load_tuning(category)
        if cached:
            cat_params = {**cat_params, **cached}
    strategy = cat_params.get("strategy", "hierarchy_blend")
    share_alpha = cat_params.get("share_alpha", 0.85)

    priors_blend = blend_dict_from_priors(cat_params)
    if "blend_weights" in cat_params:
        weights = cat_params["blend_weights"]
    else:
        weights = priors_blend

    demand_train = np.nan_to_num(df_train["Demand"].values.astype(float), nan=0.0)
    sw = np.sqrt(np.maximum(demand_train, 0) + 1)
    sw = np.where(df_train["series_cluster"].values == "stable", sw, 1.0)

    hurdle = clf = reg = None
    lgbm_direct = None

    if strategy != "segment_only" and strategy in (
        "hierarchy_blend",
        "hurdle_blend",
        "lgbm_primary",
    ) and weights.get("hurdle", 0) > 0:
        hurdle, clf, reg = train_hurdle(
            df_train, features, CAT_FEATURES, learning_rate=lr, sample_weight=sw
        )
        tune_fold = df[(df["Date"] >= "2025-10-01") & (df["Date"] < FORECAST_START)]
        hurdle.threshold = cat_params.get(
            "hurdle_threshold",
            tune_threshold(clf, reg, _prep_X(tune_fold, features), tune_fold["Demand"].values),
        )

    if strategy != "segment_only" and strategy in (
        "hierarchy_blend",
        "hierarchy_first",
        "lgbm_primary",
    ) and weights.get("lgbm", 0) > 0:
        lgbm_direct = lgb.LGBMRegressor(
            objective="regression_l1",
            n_estimators=300,
            learning_rate=lr,
            num_leaves=31,
            random_state=42,
            verbosity=-1,
        )
        lgbm_direct.fit(_prep_X(df_train, features), demand_train, categorical_feature=CAT_FEATURES)

    if tune and strategy != "segment_only" and "blend_weights" not in (params or {}):
        tune_folds: list[tuple[str, dict[str, np.ndarray], np.ndarray]] = []
        for fold_name, fold_start, fold_end in TUNING_FOLDS:
            built = _build_blend_tune_fold(
                df,
                fold_start,
                fold_end,
                features=features,
                share_alpha=share_alpha,
                lr=lr,
                bl_method=bl_method,
                hurdle=hurdle,
                lgbm_direct=lgbm_direct,
            )
            if built is not None:
                components_tune, y_tune = built
                tune_folds.append((fold_name, components_tune, y_tune))
        if tune_folds:
            weights = tune_blend_multi_fold(tune_folds, priors=priors_blend, prior_weight=0.35)
            fold_names = [n for n, _, _ in tune_folds]
            print(f"  Blend tuned on folds: {', '.join(fold_names)}")

    print(f"  strategy={strategy} blend={weights} share_alpha={share_alpha}")

    hier = HierarchicalForecaster.fit(
        df_history, share_alpha=share_alpha, learning_rate=lr
    )

    hist_for_cro = df_history.copy()

    def predict_fn(month_df: pd.DataFrame) -> np.ndarray:
        n = len(month_df)
        month = month_df["Date"].iloc[0]
        pred = np.zeros(n)

        dead = month_df["series_cluster"].values == "dead"
        tier = month_df["volume_tier"].values
        top10 = tier == "top10"
        mid = tier == "mid"
        tail = tier == "tail"

        hier_pred = hier.predict_month(month, month_df)
        lag1 = month_df["demand_lag_1"].fillna(0).values

        # LDA: segment×office forecast only, allocated by SKU shares
        if strategy == "segment_only":
            pred = hier_pred.copy()
            sparse = tail | (month_df["series_cluster"].values == "intermittent")
            pred[sparse & (lag1 <= 0)] = 0.0
            pred[dead] = 0.0
            return np.clip(pred, 0, None)

        base = np.zeros(n)
        for cl in ("stable", "intermittent"):
            m = month_df["series_cluster"].values == cl
            if m.any():
                base[m] = baseline_for_cluster(month_df.loc[m], cl, bl_method)

        components = {
            "baseline": base,
            "hierarchical": hier_pred,
        }
        if hurdle is not None:
            components["hurdle"] = hurdle.predict(_prep_X(month_df, features))
        if lgbm_direct is not None:
            components["lgbm"] = np.clip(lgbm_direct.predict(_prep_X(month_df, features)), 0, None)

        pred = blend_predictions(components, weights)

        if strategy == "hierarchy_first":
            w_h = weights.get("hierarchical", 0.7)
            sub_comp = {k: v for k, v in components.items() if k != "hierarchical"}
            sub_w = {k: weights.get(k, 0) for k in sub_comp}
            other = blend_predictions(sub_comp, sub_w)
            pred = w_h * hier_pred + (1.0 - w_h) * other

        # Tail: hierarchy-only when lag1>0, else zero (cuts false positives)
        if tail.any():
            pred[tail] = np.where(lag1[tail] > 0, hier_pred[tail], 0.0)
        if mid.any():
            pred[mid] = MID_TIER_HIER_WEIGHT * hier_pred[mid] + MID_TIER_DIRECT_WEIGHT * pred[mid]

        pred[dead] = 0.0
        return np.clip(pred, 0, None)

    df_forecast = recursive_predict(df_history, df_target, predict_fn, features)

    val = df_forecast[df_forecast["Date"] < "2026-04-01"]
    wape = compute_wape(val["Demand"].values, val["Predicted_Demand"].values)
    print(f"  Holdout WAPE (Jan–Mar 2026): {wape * 100:.2f}%")

    detail = _detailed_metrics(val, tiers)
    detail["folds"] = evaluate_folds(val)
    print(f"  By month: {', '.join(f'{k}={v}' for k, v in detail.items() if k.startswith('wape_20'))}")

    tuning_record = {
        "category": category,
        "strategy": strategy,
        "share_alpha": share_alpha,
        "blend_weights": weights,
        "hurdle_threshold": getattr(hurdle, "threshold", None),
        "baseline_method": bl_method,
    }
    save_tuning(category, {**tuning_record, "wape_pct": round(wape * 100, 4), "detail": detail})

    out = df_forecast[
        [
            "Year",
            "Month_Name",
            "Region",
            "SalesOfficeCode",
            "SalesOffice_Name",
            "Segment",
            "SKU_Code",
            "Predicted_Demand",
        ]
    ].rename(
        columns={
            "Month_Name": "Month Name",
            "SalesOfficeCode": "SalesOffice Code",
            "SalesOffice_Name": "SalesOffice Nme",
        }
    )

    metrics = {
        "category": category,
        "wape_pct": round(wape * 100, 4),
        "strategy": strategy,
        "blend_weights": weights,
        "share_alpha": share_alpha,
        "threshold": getattr(hurdle, "threshold", None),
        "baseline_method": bl_method,
        "detail": detail,
    }
    return out, metrics


def run_forecast_only(categories: list[str] | None = None, *, tune: bool = False) -> dict:
    cats = categories or CATEGORIES
    all_preds, report = [], {"team": TEAM_NAME, "categories": {}}
    for cat in cats:
        preds, metrics = forecast_category(cat, tune=tune, params=None)
        SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
        preds.to_csv(SUBMISSIONS_DIR / f"{cat}_predictions.csv", index=False)
        all_preds.append(preds)
        report["categories"][cat] = metrics
    combined = pd.concat(all_preds, ignore_index=True)
    combined.to_csv(SUBMISSION_FILE, index=False)
    wapes = [report["categories"][c]["wape_pct"] for c in cats]
    report["mean_wape_pct"] = round(sum(wapes) / len(wapes), 4)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    save_metrics(report, REPORTS_DIR / "validation_metrics.json")
    _print_summary(report, cats)
    return report


def _print_summary(report: dict, cats: list[str]) -> None:
    print(f"\n{'=' * 20} SUMMARY {'=' * 20}")
    for c in cats:
        print(f"  {c:6s}: WAPE {report['categories'][c]['wape_pct']:.2f}%")
    print(f"  Mean  : WAPE {report['mean_wape_pct']:.2f}%")
    print(f"Submission: {SUBMISSION_FILE}")


def run_pipeline(
    categories: list[str] | None = None,
    *,
    tune: bool = False,
) -> None:
    cats = categories or CATEGORIES
    print("Step 1/3: Preprocessing...")
    run_preprocess(cats)
    print("Step 2/3: Feature engineering...")
    run_features(cats)
    print("Step 3/3: Train & forecast...")
    run_forecast_only(cats, tune=tune)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    forecast_only = "--forecast-only" in sys.argv
    tune_only = "--tune-only" in sys.argv
    do_tune = "--tune" in sys.argv or tune_only

    if tune_only:
        # Skip preprocess/features; retune blends and regenerate submission
        run_forecast_only(args or None, tune=True)
    elif forecast_only:
        run_forecast_only(args or None, tune=do_tune)
    else:
        run_pipeline(args or None, tune=do_tune)
