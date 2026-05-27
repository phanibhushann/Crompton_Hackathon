"""Blend multiple forecast components to minimize WAPE."""
from __future__ import annotations

import numpy as np

from src.metrics import compute_wape


def blend_predictions(components: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    total_w = sum(weights.values())
    if total_w <= 0:
        return components[next(iter(components))]
    out = np.zeros_like(next(iter(components.values())), dtype=float)
    for name, arr in components.items():
        w = weights.get(name, 0.0) / total_w
        out += w * arr
    return np.clip(out, 0, None)


def tune_weights(
    components: dict[str, np.ndarray],
    y_true: np.ndarray,
    grid: list[dict[str, float]] | None = None,
) -> dict[str, float]:
    if grid is None:
        grid = [
            {"hurdle": 1.0, "baseline": 0.0, "hierarchical": 0.0, "lgbm": 0.0},
            {"hurdle": 0.0, "baseline": 1.0, "hierarchical": 0.0, "lgbm": 0.0},
            {"hurdle": 0.0, "baseline": 0.0, "hierarchical": 0.0, "lgbm": 1.0},
            {"hurdle": 0.5, "baseline": 0.5, "hierarchical": 0.0, "lgbm": 0.0},
            {"hurdle": 0.3, "baseline": 0.7, "hierarchical": 0.0, "lgbm": 0.0},
            {"hurdle": 0.5, "baseline": 0.0, "hierarchical": 0.0, "lgbm": 0.5},
            {"hurdle": 0.4, "baseline": 0.2, "hierarchical": 0.0, "lgbm": 0.4},
            {"hurdle": 0.6, "baseline": 0.2, "hierarchical": 0.0, "lgbm": 0.2},
        ]
    best_w, best_wape = grid[0], float("inf")
    for w in grid:
        pred = blend_predictions(components, w)
        wape = compute_wape(y_true, pred)
        if wape < best_wape:
            best_wape = wape
            best_w = w
    return best_w
