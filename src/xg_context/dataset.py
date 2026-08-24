"""Путь от сырого JSON StatsBomb к таблице ударов с признаками.

``shot.freeze_frame`` показывает только игроков, попавших в кадр.
Всех, кто был на поле, он не показывает.
Поэтому все счётчики соперников означают «сколько видно».
Число видимых сохраняется отдельной колонкой ``n_opponents_visible``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from xg_context.config import (
    GOALKEEPER_POSITION_NAME,
    KNOWN_SHOT_OUTCOMES,
    OPPONENT_RADII,
    SHOT_EVENT_TYPE,
    SPARSE_BOOLEAN_FIELDS,
)
from xg_context.features import is_penalty_shot, map_shot_outcome
from xg_context.geometry import defensive_context, goal_mouth_angle, shot_distance

logger = logging.getLogger(__name__)

__all__ = [
    "FilterLog",
    "SchemaReport",
    "add_defensive_features",
    "add_geometry_features",
    "add_target",
    "apply_shot_filters",
    "extract_shot_rows",
    "split_eligible_frames",
]


@dataclass
class FilterLog:
    """Счётчики строк на каждом шаге фильтрации.

    Каждый фильтр показывает, сколько строк удалил.
    Журнал попадает в отчёт и в `data_manifest.json`.
    """

    steps: list[dict[str, Any]] = field(default_factory=list)

    def record(self, name: str, description: str, n_before: int, n_after: int) -> None:
        self.steps.append(
            {
                "step": name,
                "description": description,
                "n_before": n_before,
                "n_after": n_after,
                "n_dropped": n_before - n_after,
                "share_dropped": round((n_before - n_after) / n_before, 6) if n_before else 0.0,
            }
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.steps)

    def log(self) -> None:
        for step in self.steps:
            logger.info(
                "Фильтр %-28s %7d -> %7d (удалено %d)",
                step["step"],
                step["n_before"],
                step["n_after"],
                step["n_dropped"],
            )


@dataclass
class SchemaReport:
    """Что реально встретилось в данных — для проверки допущений о схеме."""

    outcome_counts: dict[str, int] = field(default_factory=dict)
    unknown_outcomes: dict[str, int] = field(default_factory=dict)
    shot_type_counts: dict[str, int] = field(default_factory=dict)
    boolean_values: dict[str, dict[str, int]] = field(default_factory=dict)
    n_events_total: int = 0
    n_matches: int = 0

    @property
    def sparse_booleans_confirmed(self) -> dict[str, bool]:
        """Для каких полей подтверждено, что значение ``false`` не встречается.

        Только для таких полей отсутствие поля можно трактовать как ``False``.
        """
        return {
            name: (counts.get("False", 0) == 0)
            for name, counts in sorted(self.boolean_values.items())
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_matches": self.n_matches,
            "n_events_total": self.n_events_total,
            "outcome_counts": self.outcome_counts,
            "unknown_outcomes": self.unknown_outcomes,
            "shot_type_counts": self.shot_type_counts,
            "boolean_values": self.boolean_values,
            "sparse_booleans_confirmed": self.sparse_booleans_confirmed,
        }


def _valid_location(location: Any) -> bool:
    return (
        isinstance(location, (list, tuple))
        and len(location) >= 2
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in location[:2])
    )


def _parse_freeze_frame(freeze_frame: Any) -> dict[str, Any]:
    """Разложить `shot.freeze_frame` на координаты соперников и вратаря.

    Координаты полевых соперников возвращаются отдельно от вратаря.
    Вратарь описывается собственной геометрией.
    В счётчики плотности обороны вокруг бьющего он попадать не должен.
    """
    result: dict[str, Any] = {
        "has_freeze_frame": False,
        "opponent_x": [],
        "opponent_y": [],
        "keeper_x": float("nan"),
        "keeper_y": float("nan"),
        "n_teammates_visible": 0,
        "n_frame_invalid_locations": 0,
    }
    if not isinstance(freeze_frame, list) or not freeze_frame:
        return result

    result["has_freeze_frame"] = True
    opponent_x: list[float] = []
    opponent_y: list[float] = []
    n_teammates = 0
    n_invalid = 0

    for entry in freeze_frame:
        if not isinstance(entry, Mapping):
            n_invalid += 1
            continue
        location = entry.get("location")
        if not _valid_location(location):
            n_invalid += 1
            continue
        x, y = float(location[0]), float(location[1])

        if bool(entry.get("teammate", False)):
            n_teammates += 1
            continue

        position_name = (entry.get("position") or {}).get("name")
        if position_name == GOALKEEPER_POSITION_NAME:
            result["keeper_x"] = x
            result["keeper_y"] = y
        else:
            opponent_x.append(x)
            opponent_y.append(y)

    result["opponent_x"] = opponent_x
    result["opponent_y"] = opponent_y
    result["n_teammates_visible"] = n_teammates
    result["n_frame_invalid_locations"] = n_invalid
    return result


def extract_shot_rows(
    events: Sequence[Mapping[str, Any]],
    match_meta: Mapping[str, Any],
    schema: SchemaReport,
) -> list[dict[str, Any]]:
    """Извлечь строки ударов одного матча и обновить отчёт о схеме.

    Фильтрация здесь не выполняется.
    Возвращаются все удары, включая пенальти и удары с неизвестным исходом.
    Отбор идёт отдельным шагом с подсчётом строк.
    """
    rows: list[dict[str, Any]] = []
    schema.n_events_total += len(events)

    for event in events:
        if (event.get("type") or {}).get("name") != SHOT_EVENT_TYPE:
            continue
        shot = event.get("shot") or {}

        outcome_name = (shot.get("outcome") or {}).get("name")
        shot_type_name = (shot.get("type") or {}).get("name")
        schema.outcome_counts[str(outcome_name)] = (
            schema.outcome_counts.get(str(outcome_name), 0) + 1
        )
        schema.shot_type_counts[str(shot_type_name)] = (
            schema.shot_type_counts.get(str(shot_type_name), 0) + 1
        )
        if outcome_name not in KNOWN_SHOT_OUTCOMES:
            schema.unknown_outcomes[str(outcome_name)] = (
                schema.unknown_outcomes.get(str(outcome_name), 0) + 1
            )

        # Разреженные булевы поля: фиксируем, какие значения реально встречаются.
        for name in SPARSE_BOOLEAN_FIELDS:
            value = event.get(name, shot.get(name))
            if value is not None:
                bucket = schema.boolean_values.setdefault(name, {})
                bucket[str(bool(value))] = bucket.get(str(bool(value)), 0) + 1

        location = event.get("location")
        has_location = _valid_location(location)
        frame = _parse_freeze_frame(shot.get("freeze_frame"))

        rows.append(
            {
                "shot_id": event.get("id"),
                "match_id": int(match_meta["match_id"]),
                "competition_id": int(match_meta["competition_id"]),
                "season_id": int(match_meta["season_id"]),
                "competition_name": match_meta["competition_name"],
                "match_date": match_meta.get("match_date"),
                "period": event.get("period"),
                "minute": event.get("minute"),
                "second": event.get("second"),
                "player_id": (event.get("player") or {}).get("id"),
                "player_name": (event.get("player") or {}).get("name"),
                "team_id": (event.get("team") or {}).get("id"),
                "team_name": (event.get("team") or {}).get("name"),
                "shot_outcome": outcome_name,
                "outcome_is_known": outcome_name in KNOWN_SHOT_OUTCOMES,
                "shot_type": shot_type_name,
                "is_penalty": is_penalty_shot(shot_type_name, event.get("period")),
                "play_pattern": (event.get("play_pattern") or {}).get("name"),
                "body_part": (shot.get("body_part") or {}).get("name"),
                "shot_technique": (shot.get("technique") or {}).get("name"),
                "is_first_time": bool(shot.get("first_time", False)),
                "under_pressure": bool(event.get("under_pressure", False)),
                "one_on_one": bool(shot.get("one_on_one", False)),
                "open_goal": bool(shot.get("open_goal", False)),
                "statsbomb_xg": shot.get("statsbomb_xg"),
                "has_location": has_location,
                "shot_x": float(location[0]) if has_location else np.nan,
                "shot_y": float(location[1]) if has_location else np.nan,
                "has_freeze_frame": frame["has_freeze_frame"],
                "opponent_x": frame["opponent_x"],
                "opponent_y": frame["opponent_y"],
                "keeper_x": frame["keeper_x"],
                "keeper_y": frame["keeper_y"],
                "n_teammates_visible": frame["n_teammates_visible"],
                "n_frame_invalid_locations": frame["n_frame_invalid_locations"],
            }
        )
    return rows


def apply_shot_filters(frame: pd.DataFrame, log: FilterLog) -> pd.DataFrame:
    """Применить фильтры основной выборки, записывая счётчики строк.

    Порядок фильтров важен для интерпретации счётчиков и зафиксирован здесь.
    """
    n = len(frame)
    log.record("all_shots", "Все события типа Shot", n, n)

    before = len(frame)
    frame = frame[~frame["is_penalty"].astype(bool)]
    log.record(
        "drop_penalties",
        "Пенальти и удары серии пенальти",
        before,
        len(frame),
    )

    before = len(frame)
    frame = frame[frame["outcome_is_known"].astype(bool)]
    log.record(
        "drop_unknown_outcome",
        "Исход удара вне известной схемы StatsBomb",
        before,
        len(frame),
    )

    before = len(frame)
    frame = frame[frame["has_location"].astype(bool)]
    log.record("drop_missing_location", "Нет координат удара", before, len(frame))

    before = len(frame)
    frame = frame[frame["shot_id"].notna()]
    log.record("drop_missing_shot_id", "Нет идентификатора события", before, len(frame))

    before = len(frame)
    frame = frame.drop_duplicates(subset="shot_id", keep="first")
    log.record("drop_duplicate_shot_id", "Дубликаты shot_id", before, len(frame))

    return frame.reset_index(drop=True)


def add_target(frame: pd.DataFrame) -> pd.DataFrame:
    """Проставить бинарный target по исходу удара.

    Любой исход вне известной схемы StatsBomb поднимает исключение.
    Молча превращаться в не-гол он не должен.
    """
    frame = frame.copy()
    frame["is_goal"] = [map_shot_outcome(value) for value in frame["shot_outcome"]]
    return frame


def add_geometry_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Добавить геометрию удара: расстояние и видимый угол ворот."""
    frame = frame.copy()
    frame["shot_distance"] = shot_distance(frame["shot_x"].to_numpy(), frame["shot_y"].to_numpy())
    frame["shot_angle"] = goal_mouth_angle(frame["shot_x"].to_numpy(), frame["shot_y"].to_numpy())
    frame["shot_angle_deg"] = np.degrees(frame["shot_angle"])
    return frame


def add_defensive_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Добавить пространственный контекст обороны для каждого удара.

    Расчёт идёт построчно, потому что `freeze_frame` имеет переменную длину.
    Для нескольких десятков тысяч ударов это занимает секунды.
    """
    frame = frame.copy()
    records: list[dict[str, float]] = []
    for row in frame.itertuples(index=False):
        records.append(
            defensive_context(
                shot_x=row.shot_x,
                shot_y=row.shot_y,
                opponent_x=row.opponent_x,
                opponent_y=row.opponent_y,
                keeper_x=row.keeper_x,
                keeper_y=row.keeper_y,
                radii=OPPONENT_RADII,
            )
        )
    context = pd.DataFrame(records, index=frame.index)
    for column in context.columns:
        frame[column] = context[column]

    frame["has_goalkeeper"] = frame["has_goalkeeper"].astype(bool)
    frame["has_opponent_coordinates"] = frame["n_opponents_visible"] > 0
    return frame


def split_eligible_frames(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Разделить на ``all_eligible_shots`` и ``context_eligible_shots``.

    Контекст достаточен, если у удара есть `freeze_frame` с координатами соперника.
    Хватает одного полевого соперника.
    Наличие вратаря в это условие не входит: его отсутствие описывается признаком
    ``has_goalkeeper`` и не должно выбрасывать строку.
    """
    all_eligible = frame.reset_index(drop=True)
    mask = all_eligible["has_freeze_frame"].astype(bool) & all_eligible[
        "has_opponent_coordinates"
    ].astype(bool)
    context_eligible = all_eligible[mask].reset_index(drop=True)
    return all_eligible, context_eligible
