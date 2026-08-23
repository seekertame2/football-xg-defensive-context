"""Графики проекта с русскими подписями (спецификация, разделы 1 и 13).

Каждая функция отвечает на конкретный аналитический вопрос и подписывает
источник данных. Декоративные графики не создаются.

Единый стиль задаётся `apply_style()`; палитра подобрана так, чтобы линии
различались и в цвете, и по типу штриха.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from xg_context.config import (
    GOAL_CENTER_Y,
    GOAL_LEFT_POST_Y,
    GOAL_LINE_X,
    GOAL_RIGHT_POST_Y,
    PITCH_LENGTH,
    PITCH_WIDTH,
)

logger = logging.getLogger(__name__)

__all__ = [
    "apply_style",
    "plot_ablation",
    "plot_calibration",
    "plot_error_analysis",
    "plot_feature_importance",
    "plot_freeze_frame_examples",
    "plot_goal_rate_by_geometry",
    "plot_league_skill",
    "plot_sample_overview",
    "plot_sensitivity",
    "plot_shot_map",
    "plot_xg_surface",
    "save_figure",
]

SOURCE_NOTE = "Источник: StatsBomb Open Data, сезон 2015/2016"

#: Палитра: различима и по цвету, и по светлоте.
COLORS = {
    "geometry": "#4C72B0",
    "shot": "#DD8452",
    "flags": "#937860",
    "defensive": "#C44E52",
    "statsbomb": "#55A868",
    "neutral": "#8C8C8C",
}
LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]


def apply_style() -> None:
    """Единый стиль графиков проекта."""
    matplotlib.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.autolayout": False,
        }
    )


def save_figure(fig: plt.Figure, path: str | Path, note: str = SOURCE_NOTE) -> Path:
    """Подписать источник и сохранить рисунок."""
    if note:
        fig.text(0.01, 0.005, note, fontsize=8, color="#666666", ha="left")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target)
    plt.close(fig)
    logger.info("График сохранён: %s", target)
    return target


def _draw_attacking_third(ax: plt.Axes) -> None:
    """Нарисовать штрафную и ворота в системе координат StatsBomb."""
    ax.plot([PITCH_LENGTH, PITCH_LENGTH], [0, PITCH_WIDTH], color="#333333", lw=1.2)
    # штрафная площадь 18 ярдов
    ax.plot([102, 102], [18, 62], color="#999999", lw=1.0)
    ax.plot([102, PITCH_LENGTH], [18, 18], color="#999999", lw=1.0)
    ax.plot([102, PITCH_LENGTH], [62, 62], color="#999999", lw=1.0)
    # вратарская 6 ярдов
    ax.plot([114, 114], [30, 50], color="#BBBBBB", lw=0.9)
    ax.plot([114, PITCH_LENGTH], [30, 30], color="#BBBBBB", lw=0.9)
    ax.plot([114, PITCH_LENGTH], [50, 50], color="#BBBBBB", lw=0.9)
    # ворота
    ax.plot(
        [GOAL_LINE_X, GOAL_LINE_X],
        [GOAL_LEFT_POST_Y, GOAL_RIGHT_POST_Y],
        color="#C44E52",
        lw=3.5,
        solid_capstyle="butt",
    )
    ax.set_aspect("equal")
    ax.set_xlim(78, 122)
    ax.set_ylim(8, PITCH_WIDTH - 8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


# --------------------------------------------------------------------------------------
# Данные
# --------------------------------------------------------------------------------------


def plot_sample_overview(by_league: pd.DataFrame) -> plt.Figure:
    """Сколько матчей и ударов в каждой лиге и какова доля голов.

    Вопрос: сбалансирована ли выборка и не различаются ли лиги по базовой
    частоте голов настолько, что их нельзя считать одной популяцией?
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    order = by_league.sort_values("n_shots", ascending=False)

    axes[0].barh(order["competition_name"], order["n_matches"], color=COLORS["geometry"])
    axes[0].set_title("Матчей в лиге")
    axes[0].invert_yaxis()

    axes[1].barh(order["competition_name"], order["n_shots"], color=COLORS["shot"])
    axes[1].set_title("Непенальтистских ударов")
    axes[1].invert_yaxis()
    axes[1].set_yticklabels([])

    axes[2].barh(order["competition_name"], order["goal_rate"] * 100, color=COLORS["defensive"])
    axes[2].set_title("Доля голов, %")
    axes[2].invert_yaxis()
    axes[2].set_yticklabels([])
    for index, value in enumerate(order["goal_rate"] * 100):
        axes[2].text(value + 0.1, index, f"{value:.1f}", va="center", fontsize=9)

    fig.suptitle("Состав выборки: четыре мужские топ-лиги сезона 2015/2016", y=1.02)
    fig.tight_layout()
    return fig


def plot_shot_map(shots: pd.DataFrame, bins: int = 24) -> plt.Figure:
    """Где бьют и откуда забивают.

    Вопрос: как распределены удары по полю и где доля голов максимальна?
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].hexbin(
        shots["shot_x"],
        shots["shot_y"],
        gridsize=bins,
        cmap="Blues",
        mincnt=1,
        extent=(75, 122, 0, PITCH_WIDTH),
    )
    _draw_attacking_third(axes[0])
    axes[0].set_title("Плотность ударов")

    stat = axes[1].hexbin(
        shots["shot_x"],
        shots["shot_y"],
        C=shots["is_goal"],
        reduce_C_function=np.mean,
        gridsize=bins,
        cmap="Reds",
        mincnt=15,
        extent=(75, 122, 0, PITCH_WIDTH),
    )
    _draw_attacking_third(axes[1])
    axes[1].set_title("Доля голов (ячейки от 15 ударов)")
    fig.colorbar(stat, ax=axes[1], shrink=0.8, label="доля голов")

    fig.suptitle("Карта ударов: атака идёт вправо, ворота справа", y=1.0)
    fig.tight_layout()
    return fig


def plot_goal_rate_by_geometry(shots: pd.DataFrame, n_bins: int = 12) -> plt.Figure:
    """Как доля голов зависит от расстояния и угла.

    Вопрос: достаточно ли линейной по этим признакам модели, или связь
    существенно нелинейна?
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    for ax, column, label in (
        (axes[0], "shot_distance", "Расстояние до центра ворот, ярды"),
        (axes[1], "shot_angle_deg", "Видимый угол ворот, градусы"),
    ):
        values = shots[column]
        edges = np.quantile(values, np.linspace(0, 1, n_bins + 1))
        edges = np.unique(edges)
        index = np.clip(np.digitize(values, edges[1:-1]), 0, len(edges) - 2)
        table = (
            shots.assign(_bin=index)
            .groupby("_bin")
            .agg(centre=(column, "median"), rate=("is_goal", "mean"), n=("is_goal", "size"))
        )
        ax.plot(table["centre"], table["rate"] * 100, marker="o", color=COLORS["geometry"])
        ax.set_xlabel(label)
        ax.set_ylabel("Доля голов, %")
        ax.set_title(f"Доля голов по {'расстоянию' if 'distance' in column else 'углу'}")

    fig.suptitle("Геометрия удара объясняет многое, но связь нелинейна", y=1.02)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------------------
# Модели
# --------------------------------------------------------------------------------------


def plot_ablation(ablation: pd.DataFrame) -> plt.Figure:
    """Прирост качества по мере добавления групп признаков.

    Вопрос проекта: какую дополнительную ценность даёт САМОСТОЯТЕЛЬНО
    рассчитанный защитный контекст сверх геометрии, характеристик удара
    и готовых флагов StatsBomb?
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    levels = ["L1", "L2", "L3", "L4"]
    captions = {
        "L1": "L1\nгеометрия",
        "L2": "L2\n+ удар",
        "L3": "L3\n+ флаги SB",
        "L4": "L4\n+ оборона",
    }

    for offset, (model, group) in enumerate(ablation.groupby("модель", sort=True)):
        group = group.set_index("уровень").reindex(levels)
        color = COLORS["defensive"] if offset == 0 else COLORS["geometry"]
        axes[0].plot(
            levels,
            group["test_log_loss"],
            marker="o",
            label=model,
            color=color,
            linestyle=LINESTYLES[offset % len(LINESTYLES)],
        )
        axes[1].bar(
            np.arange(len(levels)) + offset * 0.38 - 0.19,
            -group["Δ_log_loss_шаг"].fillna(0.0),
            width=0.36,
            label=model,
            color=color,
        )

    axes[0].set_xticks(range(len(levels)))
    axes[0].set_xticklabels([captions[level] for level in levels])
    axes[0].set_ylabel("Log loss на тесте (меньше — лучше)")
    axes[0].set_title("Качество по уровням признаков")
    axes[0].legend()

    axes[1].set_xticks(range(len(levels)))
    axes[1].set_xticklabels([captions[level] for level in levels])
    axes[1].set_ylabel("Улучшение log loss на шаге")
    axes[1].set_title("Вклад каждой группы признаков")
    axes[1].axhline(0, color="#333333", lw=0.8)
    axes[1].legend()

    fig.suptitle("Ablation: одни и те же удары, одно разбиение, одна предобработка", y=1.02)
    fig.tight_layout()
    return fig


def plot_calibration(
    tables: Mapping[str, pd.DataFrame],
    *,
    title: str = "Калибровка: совпадает ли предсказанная вероятность с фактической",
) -> plt.Figure:
    """Reliability curve с размерами бинов.

    Вопрос: можно ли трактовать выход модели как настоящую вероятность гола?
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.15, 1]})

    axes[0].plot([0, 0.65], [0, 0.65], color="#333333", lw=1.0, label="идеальная калибровка")
    for index, (label, table) in enumerate(tables.items()):
        if table.empty:
            continue
        axes[0].plot(
            table["mean_predicted"],
            table["observed_rate"],
            marker="o",
            markersize=4,
            linestyle=LINESTYLES[index % len(LINESTYLES)],
            label=label,
        )
    axes[0].set_xlabel("Предсказанная вероятность гола")
    axes[0].set_ylabel("Наблюдаемая доля голов")
    axes[0].set_title("Reliability curve (бины по квантилям)")
    axes[0].legend(fontsize=9)

    first = next((t for t in tables.values() if not t.empty), None)
    if first is not None:
        axes[1].bar(range(len(first)), first["n"], color=COLORS["neutral"])
        axes[1].set_xlabel("Номер бина (от низкого xG к высокому)")
        axes[1].set_ylabel("Число ударов в бине")
        axes[1].set_title("Размеры бинов (квантильное разбиение)")

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig


def plot_feature_importance(importance: pd.DataFrame, *, top: int = 15) -> plt.Figure:
    """Коэффициенты или важности признаков с оговорками.

    Вопрос: какие признаки защитного контекста несут больше всего информации?
    """
    column = "коэффициент" if "коэффициент" in importance.columns else "важность"
    table = importance.head(top).iloc[::-1]
    colors = [
        COLORS["defensive"]
        if any(key in name for key in ("opponent", "goalkeeper", "cone", "visible"))
        else COLORS["geometry"]
        for name in table["признак"]
    ]

    fig, ax = plt.subplots(figsize=(9, 0.42 * len(table) + 1.6))
    ax.barh(table["признак"], table[column], color=colors)
    ax.axvline(0, color="#333333", lw=0.8)
    ax.set_xlabel(column.capitalize())
    ax.set_title(f"Топ-{top} признаков (красным — защитный контекст)")
    fig.tight_layout()
    return fig


def plot_xg_surface(
    predict: Any,
    template: pd.Series,
    *,
    scenarios: Sequence[tuple[str, dict[str, Any]]],
    grid_step: float = 1.0,
) -> plt.Figure:
    """Карта предсказанного xG по полю для контролируемых сценариев.

    Вопрос: как модель меняет прогноз при одном и том же положении мяча,
    но разной плотности обороны?

    Parameters
    ----------
    predict:
        Функция ``DataFrame -> вероятности``.
    template:
        Строка-шаблон: значения всех признаков, которые не варьируются.
    scenarios:
        Пары «название сценария → переопределяемые признаки».
    """
    xs = np.arange(84.0, PITCH_LENGTH + 0.001, grid_step)
    ys = np.arange(2.0, PITCH_WIDTH - 1.999, grid_step)
    mesh_x, mesh_y = np.meshgrid(xs, ys)
    flat_x, flat_y = mesh_x.ravel(), mesh_y.ravel()

    from xg_context.geometry import goal_mouth_angle, shot_distance

    fig, axes = plt.subplots(1, len(scenarios), figsize=(6.0 * len(scenarios), 5))
    axes = np.atleast_1d(axes)

    for ax, (title, overrides) in zip(axes, scenarios, strict=True):
        grid = pd.DataFrame([template.to_dict()] * len(flat_x))
        grid["shot_x"] = flat_x
        grid["shot_y"] = flat_y
        grid["shot_distance"] = shot_distance(flat_x, flat_y)
        grid["shot_angle"] = goal_mouth_angle(flat_x, flat_y)
        for key, value in overrides.items():
            grid[key] = value

        surface = predict(grid).reshape(mesh_x.shape)
        image = ax.pcolormesh(mesh_x, mesh_y, surface, cmap="magma", vmin=0.0, vmax=0.6)
        _draw_attacking_third(ax)
        ax.set_title(title)
        fig.colorbar(image, ax=ax, shrink=0.8, label="предсказанный xG")

    fig.suptitle("Карта xG при одинаковой ситуации, но разной обороне", y=1.02)
    fig.tight_layout()
    return fig


def plot_freeze_frame_examples(examples: Sequence[Mapping[str, Any]]) -> plt.Figure:
    """Разбор конкретных freeze-frame с рассчитанной геометрией.

    Вопрос: что именно видит модель в защитном контексте?
    """
    fig, axes = plt.subplots(1, len(examples), figsize=(5.6 * len(examples), 5.2))
    axes = np.atleast_1d(axes)

    for ax, example in zip(axes, examples, strict=True):
        shot_x, shot_y = example["shot_x"], example["shot_y"]
        _draw_attacking_third(ax)

        # конус удара
        ax.fill(
            [shot_x, GOAL_LINE_X, GOAL_LINE_X],
            [shot_y, GOAL_LEFT_POST_Y, GOAL_RIGHT_POST_Y],
            color=COLORS["defensive"],
            alpha=0.10,
        )
        ax.plot([shot_x, GOAL_LINE_X], [shot_y, GOAL_CENTER_Y], color="#999999", lw=0.9, ls="--")

        opponents_x = np.asarray(example.get("opponent_x", []), dtype=float)
        opponents_y = np.asarray(example.get("opponent_y", []), dtype=float)
        if opponents_x.size:
            ax.scatter(
                opponents_x,
                opponents_y,
                s=70,
                color=COLORS["defensive"],
                label="соперники",
                zorder=3,
                edgecolor="white",
                linewidth=0.8,
            )
        if np.isfinite(example.get("keeper_x", np.nan)):
            ax.scatter(
                [example["keeper_x"]],
                [example["keeper_y"]],
                s=140,
                marker="s",
                color=COLORS["statsbomb"],
                label="вратарь",
                zorder=4,
                edgecolor="white",
                linewidth=0.8,
            )
        ax.scatter(
            [shot_x],
            [shot_y],
            s=150,
            marker="*",
            color=COLORS["geometry"],
            label="бьющий",
            zorder=5,
            edgecolor="white",
            linewidth=0.8,
        )
        # Границы подгоняются под сюжет: иначе удар с фланга или из-за штрафной
        # оказывается за рамкой, и рисунок вводит в заблуждение.
        points_x = np.concatenate([[shot_x], opponents_x, [example.get("keeper_x", np.nan)]])
        points_y = np.concatenate([[shot_y], opponents_y, [example.get("keeper_y", np.nan)]])
        points_x = points_x[np.isfinite(points_x)]
        points_y = points_y[np.isfinite(points_y)]
        ax.set_xlim(min(78.0, points_x.min() - 3), 122)
        ax.set_ylim(
            max(0.0, min(8.0, points_y.min() - 3)),
            min(PITCH_WIDTH, max(PITCH_WIDTH - 8, points_y.max() + 3)),
        )

        ax.set_title(example["title"], fontsize=10)
        ax.set_xlabel(example["caption"], fontsize=9, loc="left", labelpad=8)

    axes[0].legend(loc="upper left", fontsize=9)
    fig.suptitle("Что модель видит в freeze frame", y=1.0)
    fig.tight_layout()
    return fig


def plot_error_analysis(subgroups: pd.DataFrame) -> plt.Figure:
    """Где модели ошибаются сильнее и где защитный контекст помогает больше.

    Вопрос: концентрируются ли ошибки в определённых типах ударов?
    """
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 0.34 * len(subgroups) + 2.4))
    table = subgroups.iloc[::-1]

    positions = np.arange(len(table))
    axes[0].barh(
        positions - 0.2,
        table["log_loss_без_контекста"],
        height=0.38,
        label="без защитного контекста",
        color=COLORS["flags"],
    )
    axes[0].barh(
        positions + 0.2,
        table["log_loss_с_контекстом"],
        height=0.38,
        label="с защитным контекстом",
        color=COLORS["defensive"],
    )
    axes[0].set_yticks(positions)
    axes[0].set_yticklabels(table["подгруппа"])
    axes[0].set_xlabel("Log loss (меньше — лучше)")
    axes[0].set_title("Качество по подгруппам ударов")
    axes[0].legend(fontsize=9)

    improvement = table["log_loss_без_контекста"] - table["log_loss_с_контекстом"]
    colors = [COLORS["defensive"] if v > 0 else COLORS["neutral"] for v in improvement]
    axes[1].barh(positions, improvement, color=colors)
    axes[1].axvline(0, color="#333333", lw=0.8)
    axes[1].set_yticks(positions)
    axes[1].set_yticklabels([])
    axes[1].set_xlabel("Улучшение log loss от защитного контекста")
    axes[1].set_title("Где контекст помогает сильнее")
    for index, (value, n) in enumerate(zip(improvement, table["n"], strict=True)):
        axes[1].text(value, index, f"  n={n}", va="center", fontsize=8, color="#555555")

    fig.tight_layout()
    return fig


def plot_sensitivity(bootstrap: pd.DataFrame) -> plt.Figure:
    """Устойчив ли главный эффект без спорного признака `n_opponents_visible`.

    Вопрос: не держится ли прирост от защитного контекста на признаке, который
    смешивает плотность обороны с границами поля зрения камеры?
    """
    wanted = bootstrap[
        bootstrap["сравнение"].str.contains("защитный контекст|Вучёт", regex=True)
        | bootstrap["сравнение"].str.contains("n_opponents_visible")
    ].copy()
    if wanted.empty:
        wanted = bootstrap.copy()

    labels = (
        wanted["сравнение"]
        .str.replace("Логистическая: ", "ЛР: ", regex=False)
        .str.replace("Случайный лес: ", "СЛ: ", regex=False)
        .str.replace("+ защитный контекст ", "+контекст ", regex=False)
        .str.replace("+ защитный контекст", "+контекст", regex=False)
    )
    positions = np.arange(len(wanted))[::-1]
    colors = [
        COLORS["neutral"] if "Вклад самого" in name else COLORS["defensive"]
        for name in wanted["сравнение"]
    ]

    fig, ax = plt.subplots(figsize=(11, 0.62 * len(wanted) + 2.2))
    ax.errorbar(
        wanted["delta_log_loss"],
        positions,
        xerr=[
            wanted["delta_log_loss"] - wanted["delta_log_loss_ci_low"],
            wanted["delta_log_loss_ci_high"] - wanted["delta_log_loss"],
        ],
        fmt="none",
        ecolor="#666666",
        elinewidth=1.4,
        capsize=4,
    )
    ax.scatter(wanted["delta_log_loss"], positions, s=70, c=colors, zorder=3)
    ax.axvline(0, color="#333333", lw=1.0)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Δ log loss на тесте (отрицательное — лучше), 95% интервал")
    ax.set_title("Эффект защитного контекста устойчив к удалению `n_opponents_visible`")
    fig.tight_layout()
    return fig


def plot_league_skill(league_skill: pd.DataFrame) -> plt.Figure:
    """Относительное качество внутри каждой лиги.

    Вопрос: одинаково ли модель работает в четырёх лигах, если убрать эффект
    разной базовой доли голов? Сырой log loss для этого не годится.
    """
    frame = league_skill.copy()
    order = sorted(frame["лига"].unique())
    models = [
        "Логистическая без контекста",
        "Логистическая + защитный контекст",
        "statsbomb_xg",
    ]
    models = [m for m in models if m in set(frame["модель"])]
    palette = [COLORS["flags"], COLORS["defensive"], COLORS["statsbomb"]]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    width = 0.8 / max(len(models), 1)

    for panel, (ax, column, title) in enumerate(
        (
            (axes[0], "log loss", "Сырой log loss: сравнивать лиги нельзя"),
            (axes[1], "BSS", "Brier Skill Score: сопоставимо между лигами"),
        )
    ):
        for index, model in enumerate(models):
            part = frame[frame["модель"] == model].set_index("лига").reindex(order)
            ax.bar(
                np.arange(len(order)) + index * width - 0.4 + width / 2,
                part[column],
                width=width,
                label=model,
                color=palette[index % len(palette)],
            )
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, fontsize=9)
        ax.set_title(title)
        ax.set_ylabel(column)
        if panel == 0:
            ax.legend(fontsize=9)

    rates = frame.drop_duplicates("лига").set_index("лига").reindex(order)["доля голов"]
    axes[0].set_xticklabels(
        [f"{name}\nголов {rate * 100:.1f}%" for name, rate in rates.items()], fontsize=9
    )

    fig.suptitle("Лиги различаются базовой долей голов, а не применимостью модели", y=1.02)
    fig.tight_layout()
    return fig
