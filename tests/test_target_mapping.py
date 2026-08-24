"""Тесты target mapping."""

from __future__ import annotations

import pytest

from xg_context.config import KNOWN_SHOT_OUTCOMES
from xg_context.features import (
    UnknownShotOutcomeError,
    is_penalty_shot,
    map_shot_outcome,
    map_shot_outcomes,
)


def test_goal_maps_to_one() -> None:
    assert map_shot_outcome("Goal") == 1


@pytest.mark.parametrize(
    "outcome",
    ["Saved", "Blocked", "Off T", "Post", "Wayward", "Saved Off Target", "Saved to Post"],
)
def test_valid_non_goals_map_to_zero(outcome: str) -> None:
    assert map_shot_outcome(outcome) == 0


def test_every_known_outcome_is_mapped() -> None:
    """Ни один известный исход не должен приводить к исключению."""
    values = {map_shot_outcome(outcome) for outcome in KNOWN_SHOT_OUTCOMES}
    assert values == {0, 1}


def test_missing_outcome_raises_instead_of_silent_zero() -> None:
    """Отсутствующий исход нельзя молча превращать в не-гол."""
    with pytest.raises(UnknownShotOutcomeError):
        map_shot_outcome(None)


@pytest.mark.parametrize("outcome", ["Own Goal", "goal", "GOAL", "", "Unknown"])
def test_unknown_outcome_raises(outcome: str) -> None:
    with pytest.raises(UnknownShotOutcomeError):
        map_shot_outcome(outcome)


@pytest.mark.parametrize("outcome", ["Saved Off T", "Saved To Post", "Off Target"])
def test_plausible_but_wrong_spellings_are_rejected(outcome: str) -> None:
    """Регрессия: написание исходов сверено с данными, догадки не принимаются.

    Первая версия проекта использовала "Saved Off T" и "Saved To Post"; аудит
    показал, что StatsBomb пишет "Saved Off Target" и "Saved to Post".
    """
    with pytest.raises(UnknownShotOutcomeError):
        map_shot_outcome(outcome)


def test_map_many_outcomes() -> None:
    assert map_shot_outcomes(["Goal", "Saved", "Goal", "Off T"]) == [1, 0, 1, 0]


def test_map_many_propagates_error() -> None:
    with pytest.raises(UnknownShotOutcomeError):
        map_shot_outcomes(["Goal", None])


class TestPenaltyExclusion:
    """Пенальти и удары серии исключаются из основной выборки (раздел 8.3)."""

    def test_penalty_type_is_excluded(self) -> None:
        assert is_penalty_shot("Penalty", period=2) is True

    def test_shootout_period_is_excluded(self) -> None:
        assert is_penalty_shot("Open Play", period=5) is True

    def test_open_play_is_kept(self) -> None:
        assert is_penalty_shot("Open Play", period=1) is False

    @pytest.mark.parametrize("shot_type", ["Open Play", "Free Kick", "Corner", "Kick Off"])
    def test_regular_shot_types_are_kept(self, shot_type: str) -> None:
        assert is_penalty_shot(shot_type, period=2) is False

    def test_missing_period_does_not_crash(self) -> None:
        assert is_penalty_shot("Open Play", period=None) is False
        assert is_penalty_shot("Penalty", period=None) is True
