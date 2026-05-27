"""Rolling-origin WAPE tuning for blend weights and hyperparameters."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ARTIFACTS_DIR, FORECAST_START
from src.metrics import compute_wape
from src.models.blend import blend_predictions

FoldComponents = tuple[str, dict[str, np.ndarray], np.ndarray]


def _jan_weighted_wape(scores: list[tuple[str, float]], jan_weight: float = 2.0) -> float:
    total_w, total = 0.0, 0.0
    for name, wape in scores:
        w = jan_weight if "2026" in name or "holdout" in name else 1.0
        total_w += w * wape
        total += w
    return total_w / total if total else float("inf")


def grid_blend_weights() -> list[dict[str, float]]:
    """Smaller grid for faster tuning."""
    grids = [
        {"hierarchical": 0.0, "lgbm": 1.0, "hurdle": 0.0, "baseline": 0.0},
        {"hierarchical": 0.4, "lgbm": 0.6, "hurdle": 0.0, "baseline": 0.0},
        {"hierarchical": 0.55, "lgbm": 0.45, "hurdle": 0.0, "baseline": 0.0},
        {"hierarchical": 0.7, "lgbm": 0.3, "hurdle": 0.0, "baseline": 0.0},
        {"hierarchical": 0.4, "lgbm": 0.4, "hurdle": 0.2, "baseline": 0.0},
        {"hierarchical": 0.55, "lgbm": 0.35, "hurdle": 0.1, "baseline": 0.0},
        {"hierarchical": 0.25, "lgbm": 0.5, "hurdle": 0.25, "baseline": 0.0},
        {"hierarchical": 0.0, "lgbm": 0.7, "hurdle": 0.3, "baseline": 0.0},
        {"hierarchical": 0.3, "lgbm": 0.0, "hurdle": 0.0, "baseline": 0.7},
    ]
    return grids


def _merge_with_priors(
    best_w: dict[str, float],
    priors: dict[str, float] | None,
    prior_weight: float,
) -> dict[str, float]:
    if not priors:
        return best_w
    merged = {}
    for k in set(best_w) | set(priors):
        # Data-tuned weights get higher influence; priors act as regularization
        merged[k] = round((1 - prior_weight) * best_w.get(k, 0) + prior_weight * priors.get(k, 0), 6)
    return merged


def tune_blend_multi_fold(
    folds: list[FoldComponents],
    priors: dict[str, float] | None = None,
    prior_weight: float = 0.35,
) -> dict[str, float]:
    """Grid-search blend weights; score is mean WAPE across tuning folds."""
    if not folds:
        return _merge_with_priors(
            {"lgbm": 1.0, "baseline": 0.0, "hurdle": 0.0, "hierarchical": 0.0},
            priors,
            prior_weight,
        )

    best_w: dict[str, float] | None = None
    best_score = float("inf")
    for w in grid_blend_weights():
        fold_wapes = []
        for _name, components, y_true in folds:
            pred = blend_predictions(components, w)
            fold_wapes.append(compute_wape(y_true, pred))
        avg_wape = float(np.mean(fold_wapes))
        if avg_wape < best_score:
            best_score = avg_wape
            best_w = w

    default = {"lgbm": 1.0, "baseline": 0.0, "hurdle": 0.0, "hierarchical": 0.0}
    return _merge_with_priors(best_w or default, priors, prior_weight)


def tune_blend_on_fold(
    components: dict[str, np.ndarray],
    y_true: np.ndarray,
    priors: dict[str, float] | None = None,
    prior_weight: float = 0.35,
) -> dict[str, float]:
    return tune_blend_multi_fold([("single", components, y_true)], priors, prior_weight)


def load_tuning(category: str) -> dict | None:
    path = ARTIFACTS_DIR / f"{category}_tuning.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def save_tuning(category: str, params: dict) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS_DIR / f"{category}_tuning.json"
    path.write_text(json.dumps(params, indent=2))


def default_category_params(category: str) -> dict:
    """Category priors — moderate hierarchy; LGBM-primary where it won before."""
    priors = {
        "Fans": {
            "share_alpha": 0.85,
            "w_hier_prior": 0.2,
            "w_lgbm_prior": 0.65,
            "w_hurdle_prior": 0.15,
            "hurdle_threshold": 0.5,
            "strategy": "hierarchy_blend",
        },
        "LDA": {
            "share_alpha": 0.8,
            "w_hier_prior": 0.2,
            "w_lgbm_prior": 0.7,
            "w_hurdle_prior": 0.1,
            "hurdle_threshold": 0.55,
            "strategy": "hierarchy_blend",
        },
        "Pumps": {
            "share_alpha": 0.85,
            "w_hier_prior": 0.15,
            "w_lgbm_prior": 0.85,
            "w_hurdle_prior": 0.0,
            "hurdle_threshold": 0.5,
            "strategy": "lgbm_primary",
        },
        "SDA": {
            "share_alpha": 0.85,
            "w_hier_prior": 0.2,
            "w_lgbm_prior": 0.35,
            "w_hurdle_prior": 0.45,
            "hurdle_threshold": 0.45,
            "strategy": "hurdle_blend",
        },
    }
    return priors.get(category, priors["Pumps"])


def blend_dict_from_priors(p: dict) -> dict[str, float]:
    w_h = p.get("w_hier_prior", 0.4)
    w_l = p.get("w_lgbm_prior", 0.5)
    w_u = p.get("w_hurdle_prior", 0.1)
    w_b = round(max(0.0, 1.0 - w_h - w_l - w_u), 6)
    return {"hierarchical": w_h, "lgbm": w_l, "hurdle": w_u, "baseline": w_b}
