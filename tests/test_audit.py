"""Тесты разбора ударов для аудита.

Проверка идёт на синтетических событиях в схеме StatsBomb, сеть не нужна.
Главное требование: аудит честно отличает «контекст есть» от «freeze_frame
формально присутствует, но координат соперников нет».
"""

from __future__ import annotations

from typing import Any

import pytest

from xg_context.audit import audit_match_shots, selection_bias_table, shots_to_frame


def make_shot(
    *,
    shot_id: str = "shot-1",
    outcome: str | None = "Goal",
    shot_type: str = "Open Play",
    period: int = 1,
    location: list[float] | None = None,
    freeze_frame: list[dict[str, Any]] | None = None,
    body_part: str = "Right Foot",
    statsbomb_xg: float | None = 0.12,
) -> dict[str, Any]:
    """Собрать одно событие удара в схеме StatsBomb Open Data."""
    shot: dict[str, Any] = {
        "type": {"id": 87, "name": shot_type},
        "body_part": {"id": 40, "name": body_part},
        "technique": {"id": 93, "name": "Normal"},
    }
    if outcome is not None:
        shot["outcome"] = {"id": 97, "name": outcome}
    if statsbomb_xg is not None:
        shot["statsbomb_xg"] = statsbomb_xg
    if freeze_frame is not None:
        shot["freeze_frame"] = freeze_frame

    return {
        "id": shot_id,
        "type": {"id": 16, "name": "Shot"},
        "period": period,
        "minute": 10,
        "second": 0,
        "location": location if location is not None else [108.0, 40.0],
        "play_pattern": {"id": 1, "name": "Regular Play"},
        "shot": shot,
    }


def frame_player(
    x: float, y: float, *, teammate: bool = False, position: str = "Centre Back"
) -> dict[str, Any]:
    return {
        "location": [x, y],
        "player": {"id": 1, "name": "Игрок"},
        "position": {"id": 5, "name": position},
        "teammate": teammate,
    }


GK = frame_player(118.0, 40.0, position="Goalkeeper")


def audit(events: list[dict[str, Any]]):
    records = audit_match_shots(events, match_id=1, competition_id=43, season_id=3)
    return shots_to_frame(records)


class TestShotExtraction:
    def test_only_shot_events_are_taken(self) -> None:
        events = [
            make_shot(),
            {"id": "p1", "type": {"id": 30, "name": "Pass"}, "location": [50.0, 40.0]},
        ]
        assert len(audit(events)) == 1

    def test_goal_and_non_goal_are_mapped(self) -> None:
        frame = audit(
            [make_shot(shot_id="a", outcome="Goal"), make_shot(shot_id="b", outcome="Saved")]
        )
        assert frame.set_index("shot_id")["is_goal"].to_dict() == {"a": True, "b": False}

    def test_geometry_is_computed_from_location(self) -> None:
        frame = audit([make_shot(location=[108.0, 40.0])])
        assert frame.loc[0, "shot_distance"] == pytest.approx(12.0)
        assert frame.loc[0, "shot_angle_deg"] > 0

    def test_missing_location_is_flagged_not_invented(self) -> None:
        frame = audit([make_shot(location=None) | {"location": None}])
        assert bool(frame.loc[0, "has_location"]) is False
        assert bool(frame.loc[0, "is_eligible"]) is False

    def test_unknown_outcome_is_flagged_not_silently_zero(self) -> None:
        frame = audit([make_shot(outcome="Что-то новое")])
        assert bool(frame.loc[0, "outcome_is_known"]) is False
        assert frame.loc[0, "is_goal"] is None
        assert bool(frame.loc[0, "is_eligible"]) is False

    def test_missing_outcome_is_flagged(self) -> None:
        frame = audit([make_shot(outcome=None)])
        assert bool(frame.loc[0, "outcome_is_known"]) is False
        assert bool(frame.loc[0, "is_eligible"]) is False


class TestPenaltyFiltering:
    def test_penalty_type_is_marked(self) -> None:
        frame = audit([make_shot(shot_type="Penalty")])
        assert bool(frame.loc[0, "is_penalty"]) is True
        assert bool(frame.loc[0, "is_eligible"]) is False

    def test_shootout_period_is_marked(self) -> None:
        frame = audit([make_shot(shot_type="Open Play", period=5)])
        assert bool(frame.loc[0, "is_penalty"]) is True
        assert bool(frame.loc[0, "is_eligible"]) is False

    def test_open_play_shot_is_eligible(self) -> None:
        frame = audit([make_shot()])
        assert bool(frame.loc[0, "is_penalty"]) is False
        assert bool(frame.loc[0, "is_eligible"]) is True


class TestFreezeFrameQuality:
    def test_counts_opponents_and_teammates(self) -> None:
        freeze_frame = [
            GK,
            frame_player(112.0, 38.0),
            frame_player(110.0, 44.0),
            frame_player(105.0, 41.0, teammate=True),
        ]
        frame = audit([make_shot(freeze_frame=freeze_frame)])
        row = frame.iloc[0]
        assert row["n_frame_players"] == 4
        assert row["n_opponents"] == 3
        assert row["n_teammates"] == 1
        assert bool(row["has_goalkeeper"]) is True
        assert bool(row["has_context"]) is True
        assert bool(row["has_context_with_gk"]) is True

    def test_absent_freeze_frame_means_no_context(self) -> None:
        frame = audit([make_shot(freeze_frame=None)])
        assert bool(frame.loc[0, "has_freeze_frame"]) is False
        assert bool(frame.loc[0, "has_context"]) is False

    def test_empty_freeze_frame_means_no_context(self) -> None:
        frame = audit([make_shot(freeze_frame=[])])
        assert bool(frame.loc[0, "has_freeze_frame"]) is False
        assert bool(frame.loc[0, "has_context"]) is False

    def test_frame_of_teammates_only_gives_no_defensive_context(self) -> None:
        """Формально freeze_frame есть, но защитной информации в нём нет."""
        frame = audit([make_shot(freeze_frame=[frame_player(105.0, 41.0, teammate=True)])])
        assert bool(frame.loc[0, "has_freeze_frame"]) is True
        assert bool(frame.loc[0, "has_context"]) is False

    def test_broken_coordinates_are_counted_not_imputed(self) -> None:
        freeze_frame = [
            {"location": None, "teammate": False, "position": {"name": "Centre Back"}},
            frame_player(112.0, 38.0),
        ]
        frame = audit([make_shot(freeze_frame=freeze_frame)])
        row = frame.iloc[0]
        assert row["n_invalid_locations"] == 1
        assert row["n_opponents"] == 2
        assert row["n_opponents_with_location"] == 1

    def test_goalkeeper_without_coordinates_is_not_counted_as_usable(self) -> None:
        freeze_frame = [
            {"location": None, "teammate": False, "position": {"name": "Goalkeeper"}},
            frame_player(112.0, 38.0),
        ]
        frame = audit([make_shot(freeze_frame=freeze_frame)])
        row = frame.iloc[0]
        assert bool(row["has_goalkeeper"]) is True
        assert bool(row["has_goalkeeper_location"]) is False
        assert bool(row["has_context"]) is True
        assert bool(row["has_context_with_gk"]) is False

    def test_penalty_with_freeze_frame_is_still_excluded(self) -> None:
        frame = audit([make_shot(shot_type="Penalty", freeze_frame=[GK])])
        assert bool(frame.loc[0, "has_context"]) is False


class TestSelectionBias:
    def test_compares_groups_with_and_without_context(self) -> None:
        events = [
            make_shot(shot_id="a", outcome="Goal", freeze_frame=[GK, frame_player(112.0, 38.0)]),
            make_shot(shot_id="b", outcome="Saved", freeze_frame=[GK, frame_player(112.0, 38.0)]),
            make_shot(shot_id="c", outcome="Saved", freeze_frame=None),
        ]
        table = selection_bias_table(audit(events))
        assert set(table["выборка"]) == {"с контекстом", "без контекста", "все удары"}
        assert int(table.loc[table["выборка"] == "с контекстом", "n_shots"].iloc[0]) == 2
        assert int(table.loc[table["выборка"] == "без контекста", "n_shots"].iloc[0]) == 1
        assert int(table.loc[table["выборка"] == "все удары", "n_shots"].iloc[0]) == 3

    def test_returns_empty_table_when_nothing_is_eligible(self) -> None:
        table = selection_bias_table(audit([make_shot(shot_type="Penalty")]))
        assert table.empty
