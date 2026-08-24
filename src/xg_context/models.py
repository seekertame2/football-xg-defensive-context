"""Лестница моделей от базовой частоты до защитного контекста.

Все модели собираются как `Pipeline` с общим препроцессором.
Поэтому предобработка обучается только на train-фолдах.
Наборы признаков сравниваются при одинаковой обработке данных.

Дисбаланс классов намеренно не компенсируется.
Не применяются ни SMOTE, ни oversampling, ни ``class_weight="balanced"``.
Нужна хорошо откалиброванная вероятность гола, а перевзвешивание классов её смещает.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from xg_context.config import RANDOM_SEED
from xg_context.preprocessing import build_preprocessor

logger = logging.getLogger(__name__)

__all__ = [
    "MODEL_BUILDERS",
    "build_model",
    "cross_validated_search",
    "predict_proba",
]

# Модели, которым не нужно масштабирование числовых признаков.
TREE_MODELS = frozenset({"decision_tree", "random_forest", "gradient_boosting"})


def build_model(
    name: str,
    features: Sequence[str],
    *,
    params: dict[str, Any] | None = None,
    seed: int = RANDOM_SEED,
) -> Pipeline:
    """Собрать модель лестницы по имени.

    Parameters
    ----------
    name:
        Одно из ``dummy``, ``logistic``, ``decision_tree``,
        ``random_forest``, ``gradient_boosting``.
    features:
        Набор признаков; определяет препроцессор.
    params:
        Гиперпараметры конкретной модели (без префикса ``model__``).
    """
    if name not in MODEL_BUILDERS:
        raise ValueError(f"Неизвестная модель: {name!r}. Доступны: {sorted(MODEL_BUILDERS)}.")
    params = dict(params or {})
    estimator = MODEL_BUILDERS[name](params, seed)
    preprocessor = build_preprocessor(features, scale_numeric=name not in TREE_MODELS)
    return Pipeline([("preprocess", preprocessor), ("model", estimator)])


def _dummy(params: dict[str, Any], seed: int) -> DummyClassifier:
    """Предсказывает базовую частоту голов и ничего больше."""
    return DummyClassifier(strategy=params.get("strategy", "prior"), random_state=seed)


def _logistic(params: dict[str, Any], seed: int) -> LogisticRegression:
    """Логистическая регрессия без перевзвешивания классов."""
    return LogisticRegression(
        C=params.get("C", 1.0),
        max_iter=params.get("max_iter", 2000),
        solver=params.get("solver", "lbfgs"),
        class_weight=None,
        random_state=seed,
    )


def _decision_tree(params: dict[str, Any], seed: int) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(
        max_depth=params.get("max_depth", 5),
        min_samples_leaf=params.get("min_samples_leaf", 50),
        class_weight=None,
        random_state=seed,
    )


def _random_forest(params: dict[str, Any], seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=params.get("n_estimators", 300),
        max_depth=params.get("max_depth"),
        min_samples_leaf=params.get("min_samples_leaf", 20),
        max_features=params.get("max_features", "sqrt"),
        class_weight=None,
        n_jobs=-1,
        random_state=seed,
    )


def _gradient_boosting(params: dict[str, Any], seed: int) -> HistGradientBoostingClassifier:
    """Гистограммный бустинг: быстрый и штатно работает с пропусками."""
    return HistGradientBoostingClassifier(
        learning_rate=params.get("learning_rate", 0.1),
        max_iter=params.get("max_iter", 200),
        max_leaf_nodes=params.get("max_leaf_nodes", 31),
        min_samples_leaf=params.get("min_samples_leaf", 40),
        l2_regularization=params.get("l2_regularization", 0.0),
        early_stopping=False,
        random_state=seed,
    )


MODEL_BUILDERS = {
    "dummy": _dummy,
    "logistic": _logistic,
    "decision_tree": _decision_tree,
    "random_forest": _random_forest,
    "gradient_boosting": _gradient_boosting,
}


def cross_validated_search(
    name: str,
    features: Sequence[str],
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    param_grid: dict[str, list[Any]],
    *,
    n_splits: int = 5,
    seed: int = RANDOM_SEED,
) -> tuple[Pipeline, dict[str, Any], float]:
    """Подобрать гиперпараметры внутри train с группировкой по матчам.

    `StratifiedGroupKFold` не разводит удары одного матча по разным фолдам и сохраняет долю голов.
    Метрика подбора это log loss.

    Returns
    -------
    tuple
        Обученная на всём train лучшая модель, её параметры и CV log loss.
    """
    if not param_grid:
        model = build_model(name, features, seed=seed)
        model.fit(X, y)
        return model, {}, float("nan")

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    base = build_model(name, features, seed=seed)
    grid = {f"model__{key}": values for key, values in param_grid.items()}

    search = GridSearchCV(
        base,
        param_grid=grid,
        scoring="neg_log_loss",
        cv=cv,
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )
    search.fit(X, y, groups=groups)
    best = {key.removeprefix("model__"): value for key, value in search.best_params_.items()}
    logger.info("%s: лучший log loss на CV = %.5f при %s", name, -search.best_score_, best)
    return search.best_estimator_, best, float(-search.best_score_)


def predict_proba(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Вероятность гола (класс 1) для каждой строки."""
    return model.predict_proba(X)[:, 1]
