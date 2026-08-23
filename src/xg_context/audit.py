"""Аудит доступности данных StatsBomb Open Data (этап 1 спецификации).

Модуль отвечает на вопросы раздела 6 спецификации:

* какие соревнования и сезоны доступны;
* сколько матчей и непенальтистских ударов есть;
* для какой доли ударов существует корректный ``shot.freeze_frame``;
* как часто распознаётся вратарь и присутствуют координаты соперников;
* чем выборка ударов с контекстом отличается от всех ударов;
* нужны ли отдельные файлы StatsBomb 360.

Аудит намеренно не строит модельный датасет: он собирает только те поля,
которые нужны для оценки покрытия, пропусков и возможного selection bias.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from xg_context.config import (
    GOALKEEPER_POSITION_NAME,
    KNOWN_SHOT_OUTCOMES,
    PENALTY_SHOT_TYPE,
    SHOOTOUT_PERIOD,
    SHOT_EVENT_TYPE,
)
from xg_context.data import (
    SourceInventory,
    StatsBombDownloader,
    events_path,
    three_sixty_path,
)
from xg_context.geometry import goal_mouth_angle, shot_distance

logger = logging.getLogger(__name__)

__all__ = [
    "audit_match_shots",
    "build_coverage_table",
    "context_availability_by_season",
    "freeze_frame_quality_summary",
    "probe_three_sixty",
    "sample_matches_for_audit",
    "selection_bias_table",
    "shots_to_frame",
]


# --------------------------------------------------------------------------------------
# Перепись соревнований, сезонов и матчей
# --------------------------------------------------------------------------------------


def build_coverage_table(
    competitions: Sequence[Mapping[str, Any]],
    matches_by_season: Mapping[tuple[int, int], Sequence[Mapping[str, Any]]],
    inventory: SourceInventory,
) -> pd.DataFrame:
    """Собрать полную таблицу покрытия по competition-season.

    Основана на полной переписи метаданных и git-tree источника, поэтому
    числа матчей и файлов точны, а не оценены по выборке.
    """
    rows: list[dict[str, Any]] = []
    for competition in competitions:
        competition_id = int(competition["competition_id"])
        season_id = int(competition["season_id"])
        matches = matches_by_season.get((competition_id, season_id), [])
        match_ids = [int(m["match_id"]) for m in matches]

        with_events = [m for m in match_ids if inventory.has_events(m)]
        with_360 = [m for m in match_ids if inventory.has_three_sixty(m)]

        event_bytes = inventory.total_size(events_path(m) for m in with_events)
        three_sixty_bytes = inventory.total_size(three_sixty_path(m) for m in with_360)

        dates = [str(m.get("match_date")) for m in matches if m.get("match_date")]

        rows.append(
            {
                "competition_id": competition_id,
                "season_id": season_id,
                "country_name": competition.get("country_name"),
                "competition_name": competition.get("competition_name"),
                "season_name": competition.get("season_name"),
                "gender": competition.get("competition_gender"),
                "youth": bool(competition.get("competition_youth", False)),
                "international": bool(competition.get("competition_international", False)),
                "n_matches": len(match_ids),
                "n_matches_with_events": len(with_events),
                "n_matches_with_360": len(with_360),
                "share_matches_with_360": (len(with_360) / len(match_ids)) if match_ids else 0.0,
                "events_mb": round(event_bytes / 1e6, 1),
                "three_sixty_mb": round(three_sixty_bytes / 1e6, 1),
                "match_date_min": min(dates) if dates else None,
                "match_date_max": max(dates) if dates else None,
                "declared_360_available": competition.get("match_available_360") is not None,
            }
        )

    frame = pd.DataFrame(rows)
    return frame.sort_values(["competition_name", "season_name"], ignore_index=True)


def sample_matches_for_audit(
    matches_by_season: Mapping[tuple[int, int], Sequence[Mapping[str, Any]]],
    inventory: SourceInventory,
    *,
    matches_per_season: int,
    seed: int,
    max_total: int | None = None,
) -> pd.DataFrame:
    """Детерминированно выбрать матчи для выборочного аудита событий.

    Выборка стратифицирована по competition-season: из каждого сезона берётся
    до ``matches_per_season`` матчей. Это даёт покрытие всех эпох и схем,
    не скачивая 12.8 ГБ событий целиком.
    """
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []

    for (competition_id, season_id), matches in sorted(matches_by_season.items()):
        eligible = sorted(
            int(m["match_id"]) for m in matches if inventory.has_events(int(m["match_id"]))
        )
        if not eligible:
            continue
        take = min(matches_per_season, len(eligible))
        chosen = sorted(rng.sample(eligible, take))
        for match_id in chosen:
            rows.append(
                {
                    "competition_id": competition_id,
                    "season_id": season_id,
                    "match_id": match_id,
                    "events_bytes": inventory.size_of(events_path(match_id)),
                    "has_360": inventory.has_three_sixty(match_id),
                }
            )

    frame = pd.DataFrame(rows)
    if max_total is not None and len(frame) > max_total:
        frame = frame.sample(n=max_total, random_state=seed).sort_values(
            ["competition_id", "season_id", "match_id"], ignore_index=True
        )
    return frame.reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Разбор ударов одного матча
# --------------------------------------------------------------------------------------


def _freeze_frame_stats(freeze_frame: Any) -> dict[str, Any]:
    """Посчитать состав и качество одного ``shot.freeze_frame``."""
    empty = {
        "has_freeze_frame": False,
        "n_frame_players": 0,
        "n_opponents": 0,
        "n_teammates": 0,
        "n_opponents_with_location": 0,
        "has_goalkeeper": False,
        "has_goalkeeper_location": False,
        "n_invalid_locations": 0,
    }
    if not isinstance(freeze_frame, list) or not freeze_frame:
        return empty

    n_opponents = 0
    n_teammates = 0
    n_opponents_with_location = 0
    has_goalkeeper = False
    has_goalkeeper_location = False
    n_invalid = 0

    for entry in freeze_frame:
        if not isinstance(entry, Mapping):
            n_invalid += 1
            continue
        teammate = bool(entry.get("teammate", False))
        location = entry.get("location")
        valid_location = (
            isinstance(location, (list, tuple))
            and len(location) >= 2
            and all(isinstance(v, (int, float)) for v in location[:2])
        )
        if not valid_location:
            n_invalid += 1

        position_name = (entry.get("position") or {}).get("name")
        if teammate:
            n_teammates += 1
        else:
            n_opponents += 1
            if valid_location:
                n_opponents_with_location += 1
            if position_name == GOALKEEPER_POSITION_NAME:
                has_goalkeeper = True
                if valid_location:
                    has_goalkeeper_location = True

    return {
        "has_freeze_frame": True,
        "n_frame_players": len(freeze_frame),
        "n_opponents": n_opponents,
        "n_teammates": n_teammates,
        "n_opponents_with_location": n_opponents_with_location,
        "has_goalkeeper": has_goalkeeper,
        "has_goalkeeper_location": has_goalkeeper_location,
        "n_invalid_locations": n_invalid,
    }


def audit_match_shots(
    events: Sequence[Mapping[str, Any]],
    *,
    match_id: int,
    competition_id: int,
    season_id: int,
) -> list[dict[str, Any]]:
    """Извлечь аудиторские записи по всем ударам одного матча.

    Возвращает по одной записи на удар, включая пенальти и удары серии:
    их доля тоже входит в отчёт. Фильтрация выполняется на уровне анализа.
    """
    records: list[dict[str, Any]] = []

    for event in events:
        if (event.get("type") or {}).get("name") != SHOT_EVENT_TYPE:
            continue
        shot = event.get("shot") or {}

        outcome_name = (shot.get("outcome") or {}).get("name")
        shot_type_name = (shot.get("type") or {}).get("name")
        period = event.get("period")

        location = event.get("location")
        has_location = (
            isinstance(location, (list, tuple))
            and len(location) >= 2
            and all(isinstance(v, (int, float)) for v in location[:2])
        )
        x = float(location[0]) if has_location else np.nan
        y = float(location[1]) if has_location else np.nan

        record: dict[str, Any] = {
            "match_id": match_id,
            "competition_id": competition_id,
            "season_id": season_id,
            "shot_id": event.get("id"),
            "period": period,
            "minute": event.get("minute"),
            "shot_outcome": outcome_name,
            "outcome_is_known": outcome_name in KNOWN_SHOT_OUTCOMES,
            "is_goal": (outcome_name == "Goal") if outcome_name in KNOWN_SHOT_OUTCOMES else None,
            "shot_type": shot_type_name,
            "is_penalty": shot_type_name == PENALTY_SHOT_TYPE
            or (period is not None and int(period) == SHOOTOUT_PERIOD),
            "play_pattern": (event.get("play_pattern") or {}).get("name"),
            "body_part": (shot.get("body_part") or {}).get("name"),
            "shot_technique": (shot.get("technique") or {}).get("name"),
            "has_location": has_location,
            "shot_x": x,
            "shot_y": y,
            "statsbomb_xg": shot.get("statsbomb_xg"),
            "has_statsbomb_xg": shot.get("statsbomb_xg") is not None,
            "under_pressure": event.get("under_pressure"),
            "has_under_pressure_field": "under_pressure" in event,
            "one_on_one": shot.get("one_on_one"),
            "open_goal": shot.get("open_goal"),
            "first_time": shot.get("first_time"),
            "has_key_pass": shot.get("key_pass_id") is not None,
        }

        record.update(_freeze_frame_stats(shot.get("freeze_frame")))

        if has_location:
            record["shot_distance"] = float(shot_distance(x, y))
            record["shot_angle_deg"] = float(goal_mouth_angle(x, y, degrees=True))
        else:
            record["shot_distance"] = np.nan
            record["shot_angle_deg"] = np.nan

        records.append(record)

    return records


def shots_to_frame(records: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Превратить аудиторские записи в таблицу с производными флагами."""
    frame = pd.DataFrame(list(records))
    if frame.empty:
        return frame

    # Удар пригоден для основной выборки: не пенальти, известный исход, есть координаты.
    frame["is_eligible"] = (
        ~frame["is_penalty"].astype(bool)
        & frame["outcome_is_known"].astype(bool)
        & frame["has_location"].astype(bool)
    )

    # Удар пригоден для defensive-context модели: есть freeze_frame,
    # в нём есть хотя бы один соперник с координатами.
    frame["has_context"] = (
        frame["is_eligible"]
        & frame["has_freeze_frame"].astype(bool)
        & (frame["n_opponents_with_location"] > 0)
    )

    # Более строгий вариант: дополнительно распознан вратарь с координатами.
    frame["has_context_with_gk"] = frame["has_context"] & frame["has_goalkeeper_location"].astype(
        bool
    )
    return frame


# --------------------------------------------------------------------------------------
# Агрегации для отчёта
# --------------------------------------------------------------------------------------


def context_availability_by_season(
    shots: pd.DataFrame,
    coverage: pd.DataFrame,
) -> pd.DataFrame:
    """Доступность защитного контекста в разрезе competition-season."""
    eligible = shots[shots["is_eligible"]]
    grouped = (
        eligible.groupby(["competition_id", "season_id"])
        .agg(
            n_matches_sampled=("match_id", "nunique"),
            n_shots=("shot_id", "size"),
            goal_rate=("is_goal", "mean"),
            share_with_freeze_frame=("has_freeze_frame", "mean"),
            share_with_context=("has_context", "mean"),
            share_with_gk=("has_context_with_gk", "mean"),
            median_opponents_in_frame=("n_opponents", "median"),
            share_with_statsbomb_xg=("has_statsbomb_xg", "mean"),
        )
        .reset_index()
    )
    meta = coverage[
        [
            "competition_id",
            "season_id",
            "competition_name",
            "season_name",
            "country_name",
            "gender",
            "n_matches",
            "n_matches_with_events",
            "n_matches_with_360",
            "events_mb",
        ]
    ]
    merged = grouped.merge(meta, on=["competition_id", "season_id"], how="left")
    merged["shots_per_match"] = merged["n_shots"] / merged["n_matches_sampled"]
    merged["estimated_total_shots"] = (
        (merged["shots_per_match"] * merged["n_matches_with_events"]).round().astype(int)
    )
    return merged.sort_values(["competition_name", "season_name"], ignore_index=True)


def freeze_frame_quality_summary(shots: pd.DataFrame) -> dict[str, Any]:
    """Сводка качества freeze_frame по всей выборочной аудитории ударов."""
    eligible = shots[shots["is_eligible"]]
    with_frame = eligible[eligible["has_freeze_frame"]]

    return {
        "n_shots_total": len(shots),
        "n_shots_penalty": int(shots["is_penalty"].sum()),
        "n_shots_unknown_outcome": int((~shots["outcome_is_known"]).sum()),
        "n_shots_missing_location": int((~shots["has_location"]).sum()),
        "n_shots_eligible": len(eligible),
        "n_shots_with_freeze_frame": int(eligible["has_freeze_frame"].sum()),
        "share_with_freeze_frame": float(eligible["has_freeze_frame"].mean()),
        "n_shots_with_context": int(eligible["has_context"].sum()),
        "share_with_context": float(eligible["has_context"].mean()),
        "n_shots_with_gk": int(eligible["has_context_with_gk"].sum()),
        "share_with_gk": float(eligible["has_context_with_gk"].mean()),
        "share_gk_given_freeze_frame": (
            float(with_frame["has_goalkeeper_location"].mean()) if len(with_frame) else float("nan")
        ),
        "median_opponents_in_frame": (
            float(with_frame["n_opponents"].median()) if len(with_frame) else float("nan")
        ),
        "mean_opponents_in_frame": (
            float(with_frame["n_opponents"].mean()) if len(with_frame) else float("nan")
        ),
        "n_frames_with_invalid_locations": int(with_frame["n_invalid_locations"].gt(0).sum())
        if len(with_frame)
        else 0,
        "share_with_statsbomb_xg": float(eligible["has_statsbomb_xg"].mean()),
        "goal_rate_all_eligible": float(eligible["is_goal"].mean()),
        "goal_rate_with_context": (
            float(eligible.loc[eligible["has_context"], "is_goal"].mean())
            if eligible["has_context"].any()
            else float("nan")
        ),
        "goal_rate_without_context": (
            float(eligible.loc[~eligible["has_context"], "is_goal"].mean())
            if (~eligible["has_context"]).any()
            else float("nan")
        ),
    }


def selection_bias_table(shots: pd.DataFrame) -> pd.DataFrame:
    """Сравнить удары с защитным контекстом и без него.

    Отвечает на вопрос спецификации «насколько выборка ударов с контекстом
    отличается от всех ударов». Большие расхождения означают, что
    ``context_eligible_shots`` — не случайное подмножество, и это нужно
    оговорить в выводах.
    """
    eligible = shots[shots["is_eligible"]].copy()
    if eligible.empty:
        return pd.DataFrame()

    groups = {
        "с контекстом": eligible[eligible["has_context"]],
        "без контекста": eligible[~eligible["has_context"]],
        "все удары": eligible,
    }

    rows: list[dict[str, Any]] = []
    for name, group in groups.items():
        if group.empty:
            continue
        rows.append(
            {
                "выборка": name,
                "n_shots": len(group),
                "доля голов": float(group["is_goal"].mean()),
                "медиана расстояния": float(group["shot_distance"].median()),
                "медиана угла, °": float(group["shot_angle_deg"].median()),
                "доля ударов головой": float((group["body_part"] == "Head").mean()),
                "доля open play": float((group["shot_type"] == "Open Play").mean()),
                "доля со штрафных": float((group["shot_type"] == "Free Kick").mean()),
                "медиана statsbomb_xg": float(
                    pd.to_numeric(group["statsbomb_xg"], errors="coerce").median()
                ),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Проверка StatsBomb 360
# --------------------------------------------------------------------------------------


def probe_three_sixty(
    downloader: StatsBombDownloader,
    match_id: int,
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Сравнить покрытие ударов файлом 360 и полем ``shot.freeze_frame``.

    Нужно, чтобы обоснованно ответить на вопрос спецификации: даёт ли
    отдельный источник 360 достаточно дополнительной информации, чтобы
    оправдать существенно больший объём загрузки и сложность пайплайна.
    """
    frames = downloader.load_three_sixty(match_id)
    by_event = {f.get("event_uuid"): f for f in frames if isinstance(f, Mapping)}

    shot_events = [e for e in events if (e.get("type") or {}).get("name") == SHOT_EVENT_TYPE]

    n_shots = len(shot_events)
    n_shots_in_360 = 0
    n_shots_with_ff = 0
    ff_actors = []
    sb360_actors = []
    n_visible_area = 0

    for event in shot_events:
        shot = event.get("shot") or {}
        freeze_frame = shot.get("freeze_frame")
        if isinstance(freeze_frame, list) and freeze_frame:
            n_shots_with_ff += 1
            ff_actors.append(len(freeze_frame))

        frame = by_event.get(event.get("id"))
        if frame is not None:
            n_shots_in_360 += 1
            players = frame.get("freeze_frame") or []
            sb360_actors.append(len(players))
            if frame.get("visible_area"):
                n_visible_area += 1

    sample_keys: list[str] = []
    if frames:
        first = frames[0]
        sample_keys = sorted(first.keys())
        players = first.get("freeze_frame") or []
        player_keys = sorted(players[0].keys()) if players else []
    else:
        player_keys = []

    return {
        "match_id": match_id,
        "n_360_frames_total": len(frames),
        "n_shots": n_shots,
        "n_shots_with_shot_freeze_frame": n_shots_with_ff,
        "n_shots_covered_by_360": n_shots_in_360,
        "n_shots_with_visible_area": n_visible_area,
        "mean_players_shot_freeze_frame": float(np.mean(ff_actors)) if ff_actors else float("nan"),
        "mean_players_360_frame": float(np.mean(sb360_actors)) if sb360_actors else float("nan"),
        "frame_keys": sample_keys,
        "player_keys": player_keys,
        "file_mb": round(downloader.local_path(three_sixty_path(match_id)).stat().st_size / 1e6, 1),
    }
