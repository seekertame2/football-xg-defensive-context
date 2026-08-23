"""Предобработка признаков внутри scikit-learn Pipeline (спецификация, разделы 9.5 и 10.2).

Главное правило: все преобразования — включая imputer и масштабирование —
живут внутри `Pipeline`, поэтому обучаются только на train-части и никогда
не видят target валидации или теста.

Отдельно решается вопрос пропусков в defensive-признаках. Если вратарь не
попал в кадр, его координаты неизвестны. Придумывать их нельзя, поэтому
используется медианная импутация **плюс** явный флаг ``has_goalkeeper``,
который уже есть в наборе признаков и даёт модели возможность отличить
«вратарь близко» от «вратаря не видно».
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xg_context.config import CATEGORICAL_FEATURES
from xg_context.features import assert_no_forbidden_columns

__all__ = ["build_preprocessor", "split_feature_types"]


def split_feature_types(features: Sequence[str]) -> tuple[list[str], list[str]]:
    """Разделить признаки на числовые и категориальные."""
    categorical = [f for f in features if f in CATEGORICAL_FEATURES]
    numeric = [f for f in features if f not in CATEGORICAL_FEATURES]
    return numeric, categorical


def build_preprocessor(
    features: Sequence[str],
    *,
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """Собрать `ColumnTransformer` для заданного набора признаков.

    Parameters
    ----------
    features:
        Список колонок-признаков. Проверяется на запрещённые поля до сборки:
        пайплайн обязан падать, если в матрицу попало что-то из
        `FORBIDDEN_FEATURE_COLUMNS` (спецификация, раздел 9.4).
    scale_numeric:
        Масштабировать числовые признаки. Нужно логистической регрессии,
        бесполезно деревьям — для них можно отключить.

    Raises
    ------
    ForbiddenColumnError
        Если среди признаков есть запрещённое поле.
    """
    assert_no_forbidden_columns(list(features), context="набор признаков модели")
    numeric, categorical = split_feature_types(features)

    numeric_steps: list[tuple[str, object]] = [
        ("impute", SimpleImputer(strategy="median", add_indicator=False)),
    ]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))

    transformers: list[tuple[str, object, list[str]]] = []
    if numeric:
        transformers.append(("num", Pipeline(numeric_steps), numeric))
    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(strategy="constant", fill_value="__missing__"),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", min_frequency=20),
                        ),
                    ]
                ),
                categorical,
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )


def select_features(frame: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    """Взять из таблицы только колонки-признаки, проверив отсутствие утечки."""
    missing = [f for f in features if f not in frame.columns]
    if missing:
        raise KeyError(f"В таблице нет признаков: {missing}")
    matrix = frame.loc[:, list(features)]
    assert_no_forbidden_columns(matrix, context="матрица X")
    return matrix
