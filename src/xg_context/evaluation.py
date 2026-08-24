"""Вероятностные метрики, калибровка и оценка неопределённости.

Главные метрики это log loss и калибровка.
xG оценивается как вероятностная модель, а не как классификатор.
ROC-AUC и PR-AUC идут дополнительно.

Неопределённость считается парным bootstrap по матчам, а не по отдельным ударам.
Удары одного матча зависимы, и bootstrap по строкам занизил бы доверительные интервалы.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

__all__ = [
    "calibration_table",
    "evaluate_predictions",
    "interval_excludes_zero",
    "metrics_by_group",
    "paired_bootstrap_by_match",
]

# Обрезка вероятностей: log loss обращается в бесконечность на 0 и 1.
PROBABILITY_EPS = 1e-15


def _clip(probabilities: Sequence[float] | np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(probabilities, dtype=float), PROBABILITY_EPS, 1 - PROBABILITY_EPS)


def interval_excludes_zero(low: float, high: float) -> bool:
    """Устойчива ли разница: интервал целиком по одну сторону от нуля.

    Знак значения не важен.
    Устойчиво лучше и устойчиво хуже это одинаково содержательный результат.
    Интервал через ноль означает, что различия не обнаружено.
    """
    if not np.isfinite(low) or not np.isfinite(high):
        return False
    return bool((low < 0 and high < 0) or (low > 0 and high > 0))


def evaluate_predictions(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
) -> dict[str, float]:
    """Посчитать основные и дополнительные метрики одного набора предсказаний.

    ROC-AUC и PR-AUC не определены, если в выборке один класс.
    Тогда возвращается ``NaN``, а не искусственное значение.
    """
    y_true = np.asarray(y_true, dtype=int)
    probabilities = _clip(y_prob)
    n_positive = int(y_true.sum())

    metrics: dict[str, float] = {
        "n": len(y_true),
        "n_goals": n_positive,
        "goal_rate": float(y_true.mean()) if len(y_true) else float("nan"),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, probabilities)),
        "mean_predicted": float(probabilities.mean()),
    }
    if 0 < n_positive < len(y_true):
        metrics["roc_auc"] = float(roc_auc_score(y_true, probabilities))
        metrics["pr_auc"] = float(average_precision_score(y_true, probabilities))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")
    return metrics


def calibration_table(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    *,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> pd.DataFrame:
    """Таблица калибровки: предсказанная и наблюдаемая доля голов по бинам.

    Возвращает и размеры бинов.
    Без них reliability curve нечитаема: крайние бины часто очень малы.

    Parameters
    ----------
    strategy:
        ``quantile`` — равное число ударов в бине (устойчиво при скошенном xG);
        ``uniform`` — равные интервалы вероятности.
    """
    y_true = np.asarray(y_true, dtype=int)
    probabilities = _clip(y_prob)

    if strategy == "quantile":
        edges = np.quantile(probabilities, np.linspace(0, 1, n_bins + 1))
        edges = np.unique(edges)
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)

    if len(edges) < 2:
        return pd.DataFrame()

    index = np.clip(np.digitize(probabilities, edges[1:-1], right=False), 0, len(edges) - 2)
    rows: list[dict[str, Any]] = []
    for bin_id in range(len(edges) - 1):
        mask = index == bin_id
        if not mask.any():
            continue
        rows.append(
            {
                "bin": bin_id,
                "lower": float(edges[bin_id]),
                "upper": float(edges[bin_id + 1]),
                "n": int(mask.sum()),
                "mean_predicted": float(probabilities[mask].mean()),
                "observed_rate": float(y_true[mask].mean()),
                "n_goals": int(y_true[mask].sum()),
            }
        )
    table = pd.DataFrame(rows)
    if not table.empty:
        table["gap"] = table["observed_rate"] - table["mean_predicted"]
        # ECE со взвешиванием по размеру бина — явно определённая величина.
        table.attrs["ece"] = float((table["n"] / table["n"].sum() * table["gap"].abs()).sum())
    return table


def expected_calibration_error(table: pd.DataFrame) -> float:
    """Взвешенная по размеру бина средняя абсолютная ошибка калибровки."""
    if table.empty:
        return float("nan")
    return float(
        (
            table["n"] / table["n"].sum() * (table["observed_rate"] - table["mean_predicted"]).abs()
        ).sum()
    )


def metrics_by_group(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    groups: Sequence[Any],
    *,
    group_name: str = "группа",
) -> pd.DataFrame:
    """Метрики отдельно по каждой группе (у нас — по лигам)."""
    frame = pd.DataFrame(
        {
            "y": np.asarray(y_true, dtype=int),
            "p": np.asarray(y_prob, dtype=float),
            "g": list(groups),
        }
    )
    rows: list[dict[str, Any]] = []
    for value, part in frame.groupby("g", sort=True):
        metrics = evaluate_predictions(part["y"].to_numpy(), part["p"].to_numpy())
        metrics[group_name] = value
        rows.append(metrics)
    table = pd.DataFrame(rows)
    columns = [group_name, *[c for c in table.columns if c != group_name]]
    return table[columns]


def paired_bootstrap_by_match(
    y_true: Sequence[int] | np.ndarray,
    prob_baseline: Sequence[float] | np.ndarray,
    prob_candidate: Sequence[float] | np.ndarray,
    match_ids: Sequence[Any],
    *,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Парный bootstrap разницы метрик с ресемплингом **матчей**.

    Обе модели оцениваются на одних и тех же ресемплированных матчах.
    Поэтому из разницы уходит общая дисперсия лёгких и трудных матчей.
    Остаётся именно эффект признаков.

    Отрицательная разница означает, что кандидат лучше базовой модели:
    и log loss, и Brier score — метрики «чем меньше, тем лучше».
    """
    y_true = np.asarray(y_true, dtype=int)
    baseline = _clip(prob_baseline)
    candidate = _clip(prob_candidate)
    matches = np.asarray(match_ids)

    unique_matches = np.unique(matches)
    index_by_match = {m: np.flatnonzero(matches == m) for m in unique_matches}

    def _delta(rows: np.ndarray) -> tuple[float, float]:
        y = y_true[rows]
        if y.sum() == 0 or y.sum() == len(y):
            return float("nan"), float("nan")
        delta_log = log_loss(y, candidate[rows], labels=[0, 1]) - log_loss(
            y, baseline[rows], labels=[0, 1]
        )
        delta_brier = brier_score_loss(y, candidate[rows]) - brier_score_loss(y, baseline[rows])
        return float(delta_log), float(delta_brier)

    observed_log, observed_brier = _delta(np.arange(len(y_true)))

    rng = np.random.default_rng(seed)
    deltas_log: list[float] = []
    deltas_brier: list[float] = []
    for _ in range(n_bootstrap):
        drawn = rng.choice(unique_matches, size=len(unique_matches), replace=True)
        rows = np.concatenate([index_by_match[m] for m in drawn])
        delta_log, delta_brier = _delta(rows)
        if np.isfinite(delta_log):
            deltas_log.append(delta_log)
            deltas_brier.append(delta_brier)

    alpha = (1.0 - confidence_level) / 2.0

    def _ci(values: list[float]) -> tuple[float, float]:
        if not values:
            return float("nan"), float("nan")
        return (
            float(np.quantile(values, alpha)),
            float(np.quantile(values, 1 - alpha)),
        )

    log_low, log_high = _ci(deltas_log)
    brier_low, brier_high = _ci(deltas_brier)

    return {
        "n_matches": len(unique_matches),
        "n_shots": len(y_true),
        "n_bootstrap": len(deltas_log),
        "confidence_level": confidence_level,
        "delta_log_loss": observed_log,
        "delta_log_loss_ci_low": log_low,
        "delta_log_loss_ci_high": log_high,
        "delta_log_loss_significant": interval_excludes_zero(log_low, log_high),
        "delta_brier": observed_brier,
        "delta_brier_ci_low": brier_low,
        "delta_brier_ci_high": brier_high,
        "delta_brier_significant": interval_excludes_zero(brier_low, brier_high),
    }
