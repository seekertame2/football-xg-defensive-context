"""Тесты геометрии удара.

Проверяются центральный удар, острый угол и симметрия сторон.
Отдельно проверяется удар рядом с линией ворот.
При валидных координатах не должно быть отрицательных значений и NaN.
"""

from __future__ import annotations

import numpy as np
import pytest

from xg_context.config import (
    GOAL_CENTER_Y,
    GOAL_LEFT_POST_Y,
    GOAL_LINE_X,
    GOAL_RIGHT_POST_Y,
    PITCH_LENGTH,
    PITCH_WIDTH,
)
from xg_context.geometry import (
    goal_mouth_angle,
    goalkeeper_geometry,
    point_distance,
    shot_distance,
    signed_lateral_offset,
)

PENALTY_SPOT = (108.0, 40.0)


class TestShotDistance:
    def test_distance_from_penalty_spot(self) -> None:
        assert shot_distance(*PENALTY_SPOT) == pytest.approx(12.0)

    def test_distance_at_goal_centre_is_zero(self) -> None:
        assert shot_distance(GOAL_LINE_X, GOAL_CENTER_Y) == pytest.approx(0.0)

    def test_known_diagonal_distance(self) -> None:
        assert shot_distance(117.0, 36.0) == pytest.approx(5.0)

    def test_distance_is_symmetric_across_the_pitch(self) -> None:
        left = shot_distance(100.0, GOAL_CENTER_Y - 10.0)
        right = shot_distance(100.0, GOAL_CENTER_Y + 10.0)
        assert left == pytest.approx(right)

    def test_vectorised_distance(self) -> None:
        result = shot_distance([108.0, 117.0], [40.0, 36.0])
        np.testing.assert_allclose(result, [12.0, 5.0])

    def test_point_distance_matches_pythagoras(self) -> None:
        assert point_distance(0.0, 0.0, 3.0, 4.0) == pytest.approx(5.0)


class TestGoalMouthAngle:
    def test_penalty_spot_angle(self) -> None:
        """Из точки пенальти ворота видны под известным углом 2*arctan(4/12)."""
        expected = 2.0 * np.degrees(np.arctan2(4.0, 12.0))
        assert goal_mouth_angle(*PENALTY_SPOT, degrees=True) == pytest.approx(expected)

    def test_angle_grows_when_approaching_the_goal(self) -> None:
        far = goal_mouth_angle(80.0, GOAL_CENTER_Y)
        mid = goal_mouth_angle(100.0, GOAL_CENTER_Y)
        near = goal_mouth_angle(114.0, GOAL_CENTER_Y)
        assert far < mid < near

    def test_left_right_symmetry(self) -> None:
        """Зеркальные точки видят ворота под одинаковым углом."""
        for offset in (2.0, 8.0, 15.0, 30.0):
            left = goal_mouth_angle(100.0, GOAL_CENTER_Y - offset)
            right = goal_mouth_angle(100.0, GOAL_CENTER_Y + offset)
            assert left == pytest.approx(right)

    def test_acute_angle_from_the_byline_corner(self) -> None:
        """Удар из угла у линии ворот даёт очень маленький угол."""
        acute = goal_mouth_angle(119.0, 2.0, degrees=True)
        central = goal_mouth_angle(*PENALTY_SPOT, degrees=True)
        assert 0.0 < acute < 10.0
        assert acute < central

    def test_central_shot_beats_wide_shot_at_equal_distance(self) -> None:
        """При равном расстоянии центральная позиция даёт больший угол."""
        distance = 15.0
        central = goal_mouth_angle(GOAL_LINE_X - distance, GOAL_CENTER_Y)
        wide_y = GOAL_CENTER_Y + distance
        wide = goal_mouth_angle(GOAL_LINE_X, wide_y)
        assert central > wide

    def test_on_goal_line_between_posts_is_the_limit(self) -> None:
        """Точка на линии ворот между штангами - предельный случай, угол равен pi."""
        assert goal_mouth_angle(GOAL_LINE_X, GOAL_CENTER_Y) == pytest.approx(np.pi)

    def test_on_goal_line_outside_posts_is_zero(self) -> None:
        """На линии ворот, но вне их створа, ворота не видны."""
        assert goal_mouth_angle(GOAL_LINE_X, 20.0) == pytest.approx(0.0)

    @pytest.mark.parametrize("post_y", [GOAL_LEFT_POST_Y, GOAL_RIGHT_POST_Y])
    def test_exactly_on_a_post_is_handled(self, post_y: float) -> None:
        """Вырожденный случай не даёт NaN."""
        value = goal_mouth_angle(GOAL_LINE_X, post_y)
        assert np.isfinite(value)
        assert value == pytest.approx(0.0)

    def test_degrees_flag_matches_radians(self) -> None:
        radians = goal_mouth_angle(*PENALTY_SPOT)
        degrees = goal_mouth_angle(*PENALTY_SPOT, degrees=True)
        assert degrees == pytest.approx(np.degrees(radians))


class TestNumericalRobustness:
    """При валидных координатах не должно быть ни NaN, ни отрицательных значений."""

    @staticmethod
    def _grid() -> tuple[np.ndarray, np.ndarray]:
        xs = np.linspace(60.0, PITCH_LENGTH, 41)
        ys = np.linspace(0.0, PITCH_WIDTH, 41)
        mesh_x, mesh_y = np.meshgrid(xs, ys)
        return mesh_x.ravel(), mesh_y.ravel()

    def test_angle_is_finite_and_non_negative(self) -> None:
        x, y = self._grid()
        angles = goal_mouth_angle(x, y)
        assert np.isfinite(angles).all()
        assert (angles >= 0.0).all()
        assert (angles <= np.pi + 1e-9).all()

    def test_distance_is_finite_and_non_negative(self) -> None:
        x, y = self._grid()
        distances = shot_distance(x, y)
        assert np.isfinite(distances).all()
        assert (distances >= 0.0).all()

    def test_invalid_coordinates_propagate_as_nan(self) -> None:
        """Пропущенные координаты дают NaN и не подменяются числом."""
        assert np.isnan(shot_distance(np.nan, 40.0))
        assert np.isnan(goal_mouth_angle(np.nan, 40.0))


class TestLateralOffset:
    def test_centre_has_zero_offset(self) -> None:
        assert signed_lateral_offset(GOAL_CENTER_Y) == pytest.approx(0.0)

    def test_sign_distinguishes_sides(self) -> None:
        assert signed_lateral_offset(GOAL_CENTER_Y + 5.0) > 0
        assert signed_lateral_offset(GOAL_CENTER_Y - 5.0) < 0


class TestGoalkeeperGeometry:
    def test_keeper_on_the_shot_line(self) -> None:
        result = goalkeeper_geometry(108.0, 40.0, 118.0, 40.0)
        assert result["distance_to_shot"] == pytest.approx(10.0)
        assert result["distance_to_goal_line"] == pytest.approx(2.0)
        assert result["lateral_offset"] == pytest.approx(0.0)
        assert result["distance_to_shot_goal_line"] == pytest.approx(0.0)

    def test_keeper_displaced_sideways(self) -> None:
        result = goalkeeper_geometry(108.0, 40.0, 118.0, 44.0)
        assert result["lateral_offset"] == pytest.approx(4.0)
        assert result["distance_to_shot_goal_line"] == pytest.approx(4.0)

    def test_keeper_behind_the_shooter_projects_onto_segment_end(self) -> None:
        """Проекция ограничена отрезком "бьющий - центр ворот"."""
        result = goalkeeper_geometry(108.0, 40.0, 100.0, 40.0)
        assert result["distance_to_shot"] == pytest.approx(8.0)
        assert result["distance_to_shot_goal_line"] == pytest.approx(8.0)

    def test_keeper_on_the_goal_line(self) -> None:
        result = goalkeeper_geometry(100.0, 40.0, GOAL_LINE_X, 40.0)
        assert result["distance_to_goal_line"] == pytest.approx(0.0)

    def test_missing_keeper_gives_nan_not_a_guess(self) -> None:
        """Координаты вратаря нельзя придумывать."""
        result = goalkeeper_geometry(108.0, 40.0, np.nan, np.nan)
        assert np.isnan(result["distance_to_shot"])
        assert np.isnan(result["lateral_offset"])
        assert np.isnan(result["distance_to_shot_goal_line"])

    def test_shot_from_goal_centre_has_undefined_shot_line(self) -> None:
        """Нулевой отрезок "удар - центр ворот" не должен давать деление на ноль."""
        result = goalkeeper_geometry(GOAL_LINE_X, GOAL_CENTER_Y, 119.0, 40.0)
        assert np.isnan(result["distance_to_shot_goal_line"])

    def test_vectorised_over_many_shots(self) -> None:
        result = goalkeeper_geometry([108.0, 100.0], [40.0, 40.0], [118.0, 119.0], [40.0, 40.0])
        np.testing.assert_allclose(result["distance_to_shot"], [10.0, 19.0])
        np.testing.assert_allclose(result["distance_to_goal_line"], [2.0, 1.0])
