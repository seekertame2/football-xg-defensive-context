"""Разбиение выборки с группировкой по матчам.

Удары одного матча не должны одновременно попадать в обучение и в тест.
Иначе оценка качества модели может оказаться завышенной.

Метод это жадное балансирующее назначение матчей.
Внутри лиги матчи идут в детерминированном порядке от богатых голами к бедным.
Каждый матч уходит в часть с наибольшим дефицитом ударов относительно квоты.
Так выравниваются и объём, и доля голов.
Ни случайное разбиение матчей, ни `GroupShuffleSplit` этого не дают.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "SPLIT_NAMES",
    "SplitResult",
    "load_split",
    "make_grouped_split",
    "save_split",
    "split_summary",
]

SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class SplitResult:
    """Результат разбиения: соответствие match_id -> часть выборки."""

    assignment: dict[int, str]
    sizes: dict[str, float]
    seed: int

    def series(self, match_ids: pd.Series) -> pd.Series:
        """Проставить метку части выборки каждой строке по её ``match_id``."""
        return match_ids.map(self.assignment)

    def match_ids(self, split: str) -> list[int]:
        return sorted(m for m, s in self.assignment.items() if s == split)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "sizes": self.sizes,
            "match_ids": {name: self.match_ids(name) for name in SPLIT_NAMES},
        }


def make_grouped_split(
    shots: pd.DataFrame,
    *,
    group_column: str = "match_id",
    target_column: str = "is_goal",
    stratify_column: str | None = "competition_name",
    train_size: float = 0.70,
    validation_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> SplitResult:
    """Детерминированно разбить матчи на train/validation/test.

    Parameters
    ----------
    shots:
        Таблица ударов.
        Нужны колонки ``group_column`` и ``target_column``.
        При балансировке нужна ещё ``stratify_column``.
    stratify_column:
        Колонка, доли значений которой должны совпадать во всех частях.
        У нас это лига.
        Балансировка идёт независимо внутри каждого значения.
        Поэтому доли сохраняются по построению.

    Returns
    -------
    SplitResult
        Соответствие ``match_id`` -> имя части выборки.

    Raises
    ------
    ValueError
        Если доли не суммируются в единицу или таблица пуста.
    """
    total = train_size + validation_size + test_size
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Доли частей выборки должны суммироваться в 1.0, получено {total}.")
    if shots.empty:
        raise ValueError("Пустая таблица ударов: разбивать нечего.")

    sizes = {"train": train_size, "validation": validation_size, "test": test_size}

    # Агрегируем до уровня матча: разбиение оперирует матчами, а не ударами.
    per_match = (
        shots.groupby(group_column)
        .agg(
            n_shots=(target_column, "size"),
            n_goals=(target_column, "sum"),
            **({"stratum": (stratify_column, "first")} if stratify_column is not None else {}),
        )
        .reset_index()
    )
    if stratify_column is None:
        per_match["stratum"] = "__all__"

    assignment: dict[int, str] = {}
    for stratum, group in per_match.groupby("stratum", sort=True):
        assignment.update(_assign_stratum(group, group_column, sizes, seed, str(stratum)))

    return SplitResult(assignment=assignment, sizes=sizes, seed=seed)


def _assign_stratum(
    group: pd.DataFrame,
    group_column: str,
    sizes: dict[str, float],
    seed: int,
    stratum: str,
) -> dict[int, str]:
    """Жадно распределить матчи одной лиги, выравнивая объём и долю голов."""
    # Детерминированный порядок.
    # Сначала перемешиваем с seed, зависящим от лиги, затем сортируем по числу голов.
    # Так матчи с равным числом голов расходятся по частям, а не идут подряд в одну.
    rng = random.Random(f"{seed}:{stratum}")
    order = list(group.itertuples(index=False))
    rng.shuffle(order)
    order.sort(key=lambda row: (-int(row.n_goals), -int(row.n_shots)))

    total_shots = int(group["n_shots"].sum())
    quota = {name: share * total_shots for name, share in sizes.items()}
    filled = dict.fromkeys(sizes, 0.0)
    goals = dict.fromkeys(sizes, 0.0)

    assignment: dict[int, str] = {}
    for row in order:
        # Берётся часть с наибольшим относительным дефицитом ударов.
        # При равенстве берётся часть с наименьшей текущей долей голов.
        candidate = min(
            sizes,
            key=lambda name: (
                filled[name] / quota[name] if quota[name] > 0 else float("inf"),
                goals[name] / filled[name] if filled[name] > 0 else 0.0,
                name,
            ),
        )
        assignment[int(getattr(row, group_column))] = candidate
        filled[candidate] += int(row.n_shots)
        goals[candidate] += int(row.n_goals)
    return assignment


def split_summary(
    shots: pd.DataFrame,
    split: SplitResult,
    *,
    group_column: str = "match_id",
    target_column: str = "is_goal",
    stratify_column: str | None = "competition_name",
) -> pd.DataFrame:
    """Сводка по частям выборки: объём, доля голов и состав лиг."""
    frame = shots.copy()
    frame["split"] = split.series(frame[group_column])

    rows: list[dict[str, Any]] = []
    for name in SPLIT_NAMES:
        part = frame[frame["split"] == name]
        row: dict[str, Any] = {
            "split": name,
            "n_matches": int(part[group_column].nunique()),
            "n_shots": len(part),
            "share_of_shots": float(len(part) / len(frame)) if len(frame) else 0.0,
            "n_goals": int(part[target_column].sum()),
            "goal_rate": float(part[target_column].mean()) if len(part) else float("nan"),
        }
        if stratify_column is not None:
            shares = part[stratify_column].value_counts(normalize=True)
            for value, share in shares.items():
                row[f"share_{value}"] = float(share)
        rows.append(row)
    return pd.DataFrame(rows)


def save_split(split: SplitResult, path: str | Path, shot_ids: dict[str, list[str]]) -> None:
    """Сохранить разбиение и зафиксированные shot_id тестовой части.

    Тестовые идентификаторы фиксируются один раз.
    Сохранённый список позволяет проверить, что test не использовался для выбора признаков.
    """
    payload = split.to_dict()
    payload["shot_ids"] = shot_ids
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("Разбиение сохранено: %s", target)


def load_split(path: str | Path) -> dict[str, Any]:
    """Прочитать сохранённое разбиение."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
