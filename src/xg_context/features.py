"""Target mapping и защита от leakage."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import pandas as pd

from xg_context.config import (
    EXCLUDED_SHOT_TYPES,
    FORBIDDEN_FEATURE_COLUMNS,
    GOAL_OUTCOME_NAME,
    KNOWN_SHOT_OUTCOMES,
    SHOOTOUT_PERIOD,
)

__all__ = [
    "ForbiddenColumnError",
    "UnknownShotOutcomeError",
    "assert_no_forbidden_columns",
    "is_penalty_shot",
    "map_shot_outcome",
    "map_shot_outcomes",
]


class UnknownShotOutcomeError(ValueError):
    """Исход удара отсутствует в известной схеме StatsBomb."""


class ForbiddenColumnError(ValueError):
    """В матрицу признаков попала запрещённая колонка."""


def map_shot_outcome(outcome_name: str | None) -> int:
    """Превратить исход удара StatsBomb в бинарный target.

    ``Goal`` даёт 1, любой другой **известный** исход — 0.

    Неизвестный или отсутствующий исход это ошибка данных.
    Молча превращать его в 0 нельзя: так занизилась бы доля голов и испортилась бы калибровка.

    Raises
    ------
    UnknownShotOutcomeError
        Если исход равен ``None`` или отсутствует в `KNOWN_SHOT_OUTCOMES`.
    """
    if outcome_name is None:
        raise UnknownShotOutcomeError(
            "У удара отсутствует shot.outcome — исход нельзя интерпретировать как не-гол."
        )
    if outcome_name not in KNOWN_SHOT_OUTCOMES:
        raise UnknownShotOutcomeError(
            f"Неизвестный исход удара: {outcome_name!r}. "
            f"Известные значения: {sorted(KNOWN_SHOT_OUTCOMES)}."
        )
    return int(outcome_name == GOAL_OUTCOME_NAME)


def map_shot_outcomes(outcomes: Iterable[str | None]) -> list[int]:
    """Векторная версия `map_shot_outcome` с той же строгостью."""
    return [map_shot_outcome(outcome) for outcome in outcomes]


def is_penalty_shot(shot_type_name: str | None, period: int | None = None) -> bool:
    """Проверить, что удар относится к пенальти или серии пенальти.

    Такие удары исключаются: это отдельный стандартизированный процесс.
    """
    if shot_type_name in EXCLUDED_SHOT_TYPES:
        return True
    return period is not None and int(period) == SHOOTOUT_PERIOD


def assert_no_forbidden_columns(
    columns: Sequence[str] | pd.Index | pd.DataFrame,
    *,
    forbidden: Iterable[str] = FORBIDDEN_FEATURE_COLUMNS,
    context: str = "матрица признаков",
) -> None:
    """Упасть с понятной ошибкой, если запрещённое поле попало в признаки.

    Проверяются и точные совпадения, и one-hot производные вида
    ``shot_outcome_Goal``, которые могли появиться после ``ColumnTransformer``.

    Raises
    ------
    ForbiddenColumnError
        Если найдено хотя бы одно запрещённое поле.
    """
    if isinstance(columns, pd.DataFrame):
        columns = columns.columns
    names = [str(c) for c in columns]
    forbidden_set = set(forbidden)

    hits: list[str] = []
    for name in names:
        if name in forbidden_set:
            hits.append(name)
            continue
        # one-hot и производные колонки:
        # `<forbidden>_<value>` либо `<prefix>__<forbidden>`
        bare = name.split("__")[-1]
        if bare in forbidden_set:
            hits.append(name)
            continue
        if any(bare.startswith(f"{banned}_") for banned in forbidden_set):
            hits.append(name)

    if hits:
        raise ForbiddenColumnError(
            f"Запрещённые колонки в {context}: {sorted(set(hits))}. "
            "Эти поля раскрывают исход удара либо идентичность игрока и команды."
        )


def summarize_dropped(before: int, after: int, reason: str) -> dict[str, Any]:
    """Описание одного фильтра для отчёта: сколько строк удалено и почему."""
    return {
        "reason": reason,
        "n_before": before,
        "n_after": after,
        "n_dropped": before - after,
        "share_dropped": round((before - after) / before, 6) if before else 0.0,
    }
