"""Тесты пространственных признаков защитного контекста.

Все проверки идут на синтетической геометрии с заранее известным ответом.
Отдельное внимание уделено правилу "не выдумывать координаты".
Если вратарь или соперники не попали в кадр, признак равен ``NaN`` или нулю.
Правдоподобное число вместо них недопустимо.
"""

from __future__ import annotations

import numpy as np
import pytest

from xg_context.config import GOAL_CENTER_Y, GOAL_LINE_X
from xg_context.dataset import _parse_freeze_frame
from xg_context.geometry import (
    count_opponents_within,
    defensive_context,
    nearest_opponent_distance,
    opponents_goal_side,
    point_in_shot_cone,
)

# Удар с точки пенальти: расстояние 12, ворота видны под 36.87 градуса.
SHOT = (108.0, 40.0)


class TestShotCone:
    """Конус удара - треугольник "бьющий - левая штанга - правая штанга"."""

    def test_defender_on_the_shot_line_is_inside(self) -> None:
        assert point_in_shot_cone(*SHOT, [114.0], [40.0])[0]

    def test_defender_far_to_the_side_is_outside(self) -> None:
        assert not point_in_shot_cone(*SHOT, [114.0], [10.0])[0]

    def test_defender_behind_the_shooter_is_outside(self) -> None:
        """Соперник позади бьющего не перекрывает створ."""
        assert not point_in_shot_cone(*SHOT, [100.0], [40.0])[0]

    def test_defender_behind_the_goal_line_is_outside(self) -> None:
        assert not point_in_shot_cone(*SHOT, [121.0], [40.0])[0]

    def test_cone_widens_towards_the_goal(self) -> None:
        """У линии ворот конус шире, чем у бьющего: створ 8 единиц координат."""
        near_goal = point_in_shot_cone(*SHOT, [119.5], [37.0])[0]
        near_shooter = point_in_shot_cone(*SHOT, [108.5], [37.0])[0]
        assert near_goal
        assert not near_shooter

    def test_symmetry_of_the_cone(self) -> None:
        left = point_in_shot_cone(*SHOT, [114.0], [38.0])[0]
        right = point_in_shot_cone(*SHOT, [114.0], [42.0])[0]
        assert left == right

    def test_vectorised_over_many_defenders(self) -> None:
        inside = point_in_shot_cone(*SHOT, [114.0, 114.0, 100.0], [40.0, 10.0, 40.0])
        np.testing.assert_array_equal(inside, [True, False, False])

    def test_empty_input_returns_empty(self) -> None:
        assert point_in_shot_cone(*SHOT, [], []).size == 0

    def test_degenerate_shot_from_goal_line_has_empty_cone(self) -> None:
        """С самой линии ворот треугольник схлопывается - конус пуст."""
        assert not point_in_shot_cone(GOAL_LINE_X, GOAL_CENTER_Y, [119.0], [40.0])[0]

    def test_nan_coordinates_are_not_counted_as_inside(self) -> None:
        assert not point_in_shot_cone(*SHOT, [np.nan], [np.nan])[0]


class TestNearestOpponent:
    def test_known_distance(self) -> None:
        assert nearest_opponent_distance(*SHOT, [111.0, 118.0], [44.0, 40.0]) == pytest.approx(5.0)

    def test_picks_the_closest(self) -> None:
        value = nearest_opponent_distance(*SHOT, [109.0, 118.0], [40.0, 40.0])
        assert value == pytest.approx(1.0)

    def test_no_opponents_gives_nan_not_a_large_number(self) -> None:
        """Отсутствие соперников в кадре - неизвестность, а не "очень далеко"."""
        assert np.isnan(nearest_opponent_distance(*SHOT, [], []))

    def test_only_invalid_coordinates_gives_nan(self) -> None:
        assert np.isnan(nearest_opponent_distance(*SHOT, [np.nan], [np.nan]))


class TestOpponentCounts:
    def test_counts_within_radius_inclusive(self) -> None:
        # Соперники на расстоянии 1, 2 и 5 единиц от точки удара.
        xs = [109.0, 110.0, 113.0]
        ys = [40.0, 40.0, 40.0]
        assert count_opponents_within(*SHOT, xs, ys, 1.0) == 1
        assert count_opponents_within(*SHOT, xs, ys, 2.0) == 2
        assert count_opponents_within(*SHOT, xs, ys, 5.0) == 3

    def test_counts_are_monotone_in_radius(self) -> None:
        xs = [109.0, 111.0, 115.0]
        ys = [41.0, 39.0, 40.0]
        counts = [count_opponents_within(*SHOT, xs, ys, r) for r in (1.0, 2.0, 3.0, 5.0)]
        assert counts == sorted(counts)

    def test_no_opponents_gives_zero(self) -> None:
        assert count_opponents_within(*SHOT, [], [], 5.0) == 0

    def test_goal_side_counts_only_defenders_ahead_of_the_ball(self) -> None:
        assert opponents_goal_side(108.0, [110.0, 115.0, 100.0, 108.0]) == 2

    def test_goal_side_ignores_invalid_coordinates(self) -> None:
        assert opponents_goal_side(108.0, [110.0, np.nan]) == 1


class TestDefensiveContext:
    """Полный набор признаков одного удара."""

    def test_typical_shot(self) -> None:
        context = defensive_context(
            shot_x=108.0,
            shot_y=40.0,
            opponent_x=[110.0, 114.0, 100.0],
            opponent_y=[40.0, 40.0, 40.0],
            keeper_x=118.0,
            keeper_y=40.0,
        )
        assert context["n_opponents_visible"] == 3
        assert context["nearest_opponent_distance"] == pytest.approx(2.0)
        assert context["opponents_between_shot_and_goal"] == 2
        assert context["opponents_in_shot_cone"] == 2
        assert context["has_goalkeeper"] == 1.0
        assert context["goalkeeper_distance_to_shot"] == pytest.approx(10.0)
        assert context["goalkeeper_distance_to_goal_line"] == pytest.approx(2.0)
        assert context["goalkeeper_lateral_offset"] == pytest.approx(0.0)
        assert context["goalkeeper_in_shot_cone"] == 1.0

    def test_all_radius_counters_present(self) -> None:
        context = defensive_context(108.0, 40.0, [109.0], [40.0], 118.0, 40.0)
        for radius in (1, 2, 3, 5):
            assert f"opponents_within_{radius}y" in context

    def test_missing_goalkeeper_gives_nan_not_invented_coordinates(self) -> None:
        """Ключевое требование раздела 9.5: координаты вратаря не выдумываются."""
        context = defensive_context(108.0, 40.0, [110.0], [40.0])
        assert context["has_goalkeeper"] == 0.0
        assert np.isnan(context["goalkeeper_distance_to_shot"])
        assert np.isnan(context["goalkeeper_distance_to_goal_line"])
        assert np.isnan(context["goalkeeper_lateral_offset"])
        assert np.isnan(context["goalkeeper_in_shot_cone"])

    def test_missing_goalkeeper_does_not_break_opponent_features(self) -> None:
        context = defensive_context(108.0, 40.0, [110.0, 114.0], [40.0, 40.0])
        assert context["n_opponents_visible"] == 2
        assert context["opponents_in_shot_cone"] == 2

    def test_empty_frame_gives_nan_distance_and_zero_counts(self) -> None:
        context = defensive_context(108.0, 40.0, [], [])
        assert context["n_opponents_visible"] == 0
        assert np.isnan(context["nearest_opponent_distance"])
        assert context["opponents_in_shot_cone"] == 0
        assert context["opponents_between_shot_and_goal"] == 0

    def test_invalid_opponent_coordinates_are_dropped_not_imputed(self) -> None:
        context = defensive_context(108.0, 40.0, [110.0, np.nan], [40.0, np.nan], 118.0, 40.0)
        assert context["n_opponents_visible"] == 1

    def test_counters_never_exceed_visible_opponents(self) -> None:
        """Число видимых соперников - верхняя граница для всех счётчиков."""
        context = defensive_context(
            100.0, 30.0, [105.0, 110.0, 115.0, 118.0], [32.0, 35.0, 38.0, 40.0], 119.0, 40.0
        )
        visible = context["n_opponents_visible"]
        assert context["opponents_in_shot_cone"] <= visible
        assert context["opponents_between_shot_and_goal"] <= visible
        assert context["opponents_within_5y"] <= visible


class TestFreezeFrameParsing:
    """Разбор `shot.freeze_frame`: вратарь отделяется от полевых соперников."""

    @staticmethod
    def player(x, y, *, teammate=False, position="Centre Back"):
        return {
            "location": [x, y],
            "teammate": teammate,
            "position": {"name": position},
            "player": {"id": 1, "name": "Игрок"},
        }

    def test_splits_opponents_teammates_and_keeper(self) -> None:
        frame = _parse_freeze_frame(
            [
                self.player(118.0, 40.0, position="Goalkeeper"),
                self.player(112.0, 38.0),
                self.player(110.0, 44.0),
                self.player(105.0, 41.0, teammate=True),
            ]
        )
        assert frame["has_freeze_frame"] is True
        assert frame["opponent_x"] == [112.0, 110.0]
        assert frame["keeper_x"] == 118.0
        assert frame["keeper_y"] == 40.0
        assert frame["n_teammates_visible"] == 1

    def test_keeper_is_excluded_from_field_opponents(self) -> None:
        """Вратарь не должен попадать в счётчики плотности обороны."""
        frame = _parse_freeze_frame([self.player(118.0, 40.0, position="Goalkeeper")])
        assert frame["opponent_x"] == []
        assert frame["keeper_x"] == 118.0

    def test_absent_frame(self) -> None:
        frame = _parse_freeze_frame(None)
        assert frame["has_freeze_frame"] is False
        assert frame["opponent_x"] == []
        assert np.isnan(frame["keeper_x"])

    def test_empty_frame(self) -> None:
        frame = _parse_freeze_frame([])
        assert frame["has_freeze_frame"] is False

    def test_broken_locations_are_counted(self) -> None:
        frame = _parse_freeze_frame(
            [
                {"location": None, "teammate": False, "position": {"name": "Centre Back"}},
                self.player(112.0, 38.0),
            ]
        )
        assert frame["n_frame_invalid_locations"] == 1
        assert frame["opponent_x"] == [112.0]

    def test_keeper_without_coordinates_stays_unknown(self) -> None:
        frame = _parse_freeze_frame(
            [{"location": None, "teammate": False, "position": {"name": "Goalkeeper"}}]
        )
        assert np.isnan(frame["keeper_x"])
        assert frame["n_frame_invalid_locations"] == 1

    def test_teammate_goalkeeper_is_not_taken_as_opponent_keeper(self) -> None:
        """Свой вратарь в кадре не должен подменять вратаря соперника."""
        frame = _parse_freeze_frame([self.player(10.0, 40.0, teammate=True, position="Goalkeeper")])
        assert np.isnan(frame["keeper_x"])
        assert frame["n_teammates_visible"] == 1
