"""Геометрия удара в системе координат StatsBomb.

Все функции — чистые и векторизованные по NumPy, чтобы их можно было
покрыть тестами и переиспользовать в ноутбуках без дублирования логики
(спецификация, разделы 9.1 и 18).

Система координат: поле 120 x 80, ворота соперника на линии ``x = 120``
между ``y = 36`` и ``y = 44``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from xg_context.config import (
    GEOMETRY_EPS,
    GOAL_CENTER_Y,
    GOAL_LEFT_POST_Y,
    GOAL_LINE_X,
    GOAL_RIGHT_POST_Y,
)

__all__ = [
    "count_opponents_within",
    "defensive_context",
    "goal_mouth_angle",
    "goalkeeper_geometry",
    "nearest_opponent_distance",
    "opponents_goal_side",
    "point_distance",
    "point_in_shot_cone",
    "shot_distance",
    "signed_lateral_offset",
]


def _as_float_array(values: ArrayLike) -> NDArray[np.float64]:
    """Привести вход к массиву float с NaN вместо None."""
    return np.asarray(values, dtype=np.float64)


def point_distance(
    x1: ArrayLike,
    y1: ArrayLike,
    x2: ArrayLike,
    y2: ArrayLike,
) -> NDArray[np.float64]:
    """Евклидово расстояние между двумя точками поля."""
    dx = _as_float_array(x1) - _as_float_array(x2)
    dy = _as_float_array(y1) - _as_float_array(y2)
    return np.hypot(dx, dy)


def shot_distance(x: ArrayLike, y: ArrayLike) -> NDArray[np.float64]:
    """Расстояние от точки удара до центра ворот соперника."""
    return point_distance(x, y, GOAL_LINE_X, GOAL_CENTER_Y)


def goal_mouth_angle(x: ArrayLike, y: ArrayLike, *, degrees: bool = False) -> NDArray[np.float64]:
    """Видимый угол ворот из точки удара.

    Угол между направлениями на две штанги. Считается через ``arctan2`` от
    векторного и скалярного произведения векторов «точка → штанга»: эта форма
    численно устойчива и на острых углах, и почти на линии ворот, в отличие от
    разности арктангенсов или ``arccos`` от нормированного скалярного произведения.

    Возвращает значение в ``[0, pi)`` радиан (или в градусах при ``degrees=True``).
    Для точек ровно на линии ворот между штангами угол равен ``pi`` в пределе;
    вне отрезка ворот на той же линии — ``0``.

    Parameters
    ----------
    x, y:
        Координаты удара в системе StatsBomb.
    degrees:
        Вернуть результат в градусах вместо радиан.
    """
    x_arr = _as_float_array(x)
    y_arr = _as_float_array(y)

    # Векторы из точки удара в левую и правую штангу.
    left_dx = GOAL_LINE_X - x_arr
    left_dy = GOAL_LEFT_POST_Y - y_arr
    right_dx = GOAL_LINE_X - x_arr
    right_dy = GOAL_RIGHT_POST_Y - y_arr

    cross = left_dx * right_dy - left_dy * right_dx
    dot = left_dx * right_dx + left_dy * right_dy

    angle = np.arctan2(np.abs(cross), dot)

    # Вырожденный случай: точка совпадает со штангой — вектор нулевой,
    # arctan2(0, 0) вернул бы 0. Такие удары геометрически неинформативны.
    degenerate = (np.hypot(left_dx, left_dy) < GEOMETRY_EPS) | (
        np.hypot(right_dx, right_dy) < GEOMETRY_EPS
    )
    angle = np.where(degenerate, 0.0, angle)

    if degrees:
        return np.degrees(angle)
    return angle


def signed_lateral_offset(y: ArrayLike) -> NDArray[np.float64]:
    """Боковое смещение точки относительно центра ворот.

    Положительное значение — сторона правой штанги (``y > 40``).
    """
    return _as_float_array(y) - GOAL_CENTER_Y


def goalkeeper_geometry(
    shot_x: ArrayLike,
    shot_y: ArrayLike,
    keeper_x: ArrayLike,
    keeper_y: ArrayLike,
) -> dict[str, NDArray[np.float64]]:
    """Базовая геометрия вратаря относительно удара и ворот.

    Возвращает словарь с признаками:

    ``distance_to_shot``
        расстояние от вратаря до бьющего;
    ``distance_to_goal_line``
        насколько вратарь вышел из ворот вдоль оси ``x``;
    ``lateral_offset``
        боковое смещение вратаря относительно центра ворот;
    ``distance_to_shot_goal_line``
        расстояние от вратаря до отрезка «бьющий — центр ворот»;
        малое значение означает, что вратарь стоит на линии удара.

    Отсутствующие координаты вратаря дают ``NaN`` и не подменяются числами
    (спецификация, раздел 9.5).
    """
    sx = _as_float_array(shot_x)
    sy = _as_float_array(shot_y)
    kx = _as_float_array(keeper_x)
    ky = _as_float_array(keeper_y)

    distance_to_shot = point_distance(sx, sy, kx, ky)
    distance_to_goal_line = GOAL_LINE_X - kx
    lateral_offset = signed_lateral_offset(ky)

    # Расстояние от вратаря до отрезка «удар → центр ворот».
    seg_dx = GOAL_LINE_X - sx
    seg_dy = GOAL_CENTER_Y - sy
    seg_len_sq = seg_dx**2 + seg_dy**2

    safe_len_sq = np.where(seg_len_sq < GEOMETRY_EPS, 1.0, seg_len_sq)
    t = ((kx - sx) * seg_dx + (ky - sy) * seg_dy) / safe_len_sq
    t = np.clip(t, 0.0, 1.0)
    proj_x = sx + t * seg_dx
    proj_y = sy + t * seg_dy
    distance_to_line = point_distance(kx, ky, proj_x, proj_y)
    distance_to_line = np.where(seg_len_sq < GEOMETRY_EPS, np.nan, distance_to_line)

    return {
        "distance_to_shot": distance_to_shot,
        "distance_to_goal_line": distance_to_goal_line,
        "lateral_offset": lateral_offset,
        "distance_to_shot_goal_line": distance_to_line,
    }


# --------------------------------------------------------------------------------------
# Пространственный контекст защитников
#
# Все функции работают с координатами соперников одного удара. `freeze_frame`
# показывает только игроков, попавших в кадр, поэтому любой счётчик здесь —
# это «сколько соперников ВИДНО», а не «сколько их на поле». Это ограничение
# явно фиксируется признаком `n_opponents_visible` и оговаривается в отчёте.
# --------------------------------------------------------------------------------------


def point_in_shot_cone(
    shot_x: float,
    shot_y: float,
    opponent_x: ArrayLike,
    opponent_y: ArrayLike,
) -> NDArray[np.bool_]:
    """Находится ли точка внутри треугольника «удар — левая штанга — правая штанга».

    Это и есть «конус удара»: соперник внутри него физически перекрывает часть
    створа, видимого бьющему.

    Проверка — через знаки векторных произведений для трёх рёбер треугольника.
    Форма устойчива к ориентации треугольника (бьющий может быть слева или справа
    от центра) и не требует деления.

    Вырожденный случай — удар с самой линии ворот, треугольник схлопывается:
    тогда конус пуст и функция возвращает ``False`` для всех точек.
    """
    px = _as_float_array(opponent_x)
    py = _as_float_array(opponent_y)

    ax, ay = float(shot_x), float(shot_y)
    bx, by = GOAL_LINE_X, GOAL_LEFT_POST_Y
    cx, cy = GOAL_LINE_X, GOAL_RIGHT_POST_Y

    area2 = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    if abs(area2) < GEOMETRY_EPS:
        return np.zeros(px.shape, dtype=bool)

    d1 = (px - ax) * (by - ay) - (py - ay) * (bx - ax)
    d2 = (px - bx) * (cy - by) - (py - by) * (cx - bx)
    d3 = (px - cx) * (ay - cy) - (py - cy) * (ax - cx)

    has_negative = (d1 < 0) | (d2 < 0) | (d3 < 0)
    has_positive = (d1 > 0) | (d2 > 0) | (d3 > 0)
    inside = ~(has_negative & has_positive)
    return np.asarray(inside & np.isfinite(px) & np.isfinite(py), dtype=bool)


def nearest_opponent_distance(
    shot_x: float,
    shot_y: float,
    opponent_x: ArrayLike,
    opponent_y: ArrayLike,
) -> float:
    """Расстояние до ближайшего видимого соперника.

    Возвращает ``NaN``, если в кадре нет ни одного соперника с координатами:
    придумывать расстояние нельзя (спецификация, раздел 9.5).
    """
    distances = point_distance(shot_x, shot_y, opponent_x, opponent_y)
    distances = distances[np.isfinite(distances)]
    if distances.size == 0:
        return float("nan")
    return float(distances.min())


def count_opponents_within(
    shot_x: float,
    shot_y: float,
    opponent_x: ArrayLike,
    opponent_y: ArrayLike,
    radius: float,
) -> int:
    """Сколько видимых соперников находится не дальше ``radius`` ярдов от бьющего."""
    distances = point_distance(shot_x, shot_y, opponent_x, opponent_y)
    return int(np.sum(np.isfinite(distances) & (distances <= radius)))


def opponents_goal_side(
    shot_x: float,
    opponent_x: ArrayLike,
) -> int:
    """Сколько видимых соперников находится между бьющим и линией ворот.

    Определение простое и интерпретируемое: соперник ближе к воротам по оси ``x``,
    чем бьющий. В отличие от конуса удара, эта величина не требует попадания в
    створ и описывает общую плотность обороны перед мячом.
    """
    px = _as_float_array(opponent_x)
    return int(np.sum(np.isfinite(px) & (px > float(shot_x))))


def defensive_context(
    shot_x: float,
    shot_y: float,
    opponent_x: ArrayLike,
    opponent_y: ArrayLike,
    keeper_x: float = float("nan"),
    keeper_y: float = float("nan"),
    radii: tuple[float, ...] = (1.0, 2.0, 3.0, 5.0),
) -> dict[str, float]:
    """Полный набор defensive-признаков одного удара.

    Parameters
    ----------
    shot_x, shot_y:
        Координаты удара.
    opponent_x, opponent_y:
        Координаты **полевых** соперников из `freeze_frame` (без вратаря).
    keeper_x, keeper_y:
        Координаты вратаря соперника; ``NaN``, если он не попал в кадр.
    radii:
        Радиусы для счётчиков плотности обороны.

    Notes
    -----
    Все счётчики считают только видимых в кадре игроков. Отсутствие вратаря
    даёт ``NaN`` во всех вратарских признаках и ``0`` в флаге ``has_goalkeeper``:
    координаты не выдумываются.
    """
    px = _as_float_array(opponent_x)
    py = _as_float_array(opponent_y)
    valid = np.isfinite(px) & np.isfinite(py)
    px, py = px[valid], py[valid]

    in_cone = point_in_shot_cone(shot_x, shot_y, px, py)

    features: dict[str, float] = {
        "n_opponents_visible": float(px.size),
        "nearest_opponent_distance": nearest_opponent_distance(shot_x, shot_y, px, py),
        "opponents_in_shot_cone": float(np.sum(in_cone)),
        "opponents_between_shot_and_goal": float(opponents_goal_side(shot_x, px)),
    }
    for radius in radii:
        key = f"opponents_within_{radius:g}y".replace(".", "_")
        features[key] = float(count_opponents_within(shot_x, shot_y, px, py, radius))

    has_keeper = bool(np.isfinite(keeper_x) and np.isfinite(keeper_y))
    features["has_goalkeeper"] = float(has_keeper)
    if has_keeper:
        keeper = goalkeeper_geometry(shot_x, shot_y, keeper_x, keeper_y)
        features["goalkeeper_distance_to_shot"] = float(keeper["distance_to_shot"])
        features["goalkeeper_distance_to_goal_line"] = float(keeper["distance_to_goal_line"])
        features["goalkeeper_lateral_offset"] = float(keeper["lateral_offset"])
        features["goalkeeper_distance_to_shot_line"] = float(keeper["distance_to_shot_goal_line"])
        features["goalkeeper_in_shot_cone"] = float(
            point_in_shot_cone(shot_x, shot_y, [keeper_x], [keeper_y])[0]
        )
    else:
        features["goalkeeper_distance_to_shot"] = float("nan")
        features["goalkeeper_distance_to_goal_line"] = float("nan")
        features["goalkeeper_lateral_offset"] = float("nan")
        features["goalkeeper_distance_to_shot_line"] = float("nan")
        features["goalkeeper_in_shot_cone"] = float("nan")
    return features
