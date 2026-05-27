"""Two-stage hurdle model: P(demand>0) + L1 regressor on positives."""
from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import BASE_FEATURES, CAT_FEATURES


@dataclass
class HurdleModel:
    classifier: lgb.LGBMClassifier
    regressor: lgb.LGBMRegressor
    threshold: float = 0.5

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        prob = self.classifier.predict_proba(X)[:, 1]
        reg = np.clip(self.regressor.predict(X), 0, None)
        return np.where(prob >= self.threshold, reg, 0.0)


def train_hurdle(
    df_train: pd.DataFrame,
    features: list[str],
    cat_features: list[str],
    *,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    sample_weight: np.ndarray | None = None,
) -> tuple[HurdleModel, lgb.LGBMClassifier, lgb.LGBMRegressor]:
    X = df_train[features].copy()
    for col in cat_features:
        X[col] = X[col].astype("category")

    y = df_train["Demand"].values
    y_bin = (y > 0).astype(int)

    clf = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        random_state=42,
        verbosity=-1,
        class_weight="balanced",
    )
    clf.fit(X, y_bin, categorical_feature=cat_features)

    pos_mask = y > 0
    reg = lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        random_state=42,
        verbosity=-1,
    )
    sw = sample_weight[pos_mask] if sample_weight is not None else None
    reg.fit(X.loc[pos_mask], y[pos_mask], categorical_feature=cat_features, sample_weight=sw)

    return HurdleModel(classifier=clf, regressor=reg), clf, reg


def tune_threshold(
    clf: lgb.LGBMClassifier,
    reg: lgb.LGBMRegressor,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    thresholds: list[float] | None = None,
) -> float:
    from src.metrics import compute_wape

    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.3, 0.95, 0.05)]

    prob = clf.predict_proba(X_val)[:, 1]
    reg_pred = np.clip(reg.predict(X_val), 0, None)
    best_t, best_wape = 0.5, float("inf")
    for t in thresholds:
        pred = np.where(prob >= t, reg_pred, 0.0)
        w = compute_wape(y_val, pred)
        if w < best_wape:
            best_wape = w
            best_t = t
    return best_t
