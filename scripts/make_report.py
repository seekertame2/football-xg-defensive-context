"""Графики, анализ ошибок и итоговый отчёт (этап 5 спецификации).

Скрипт читает результаты `scripts/run_experiments.py` и не переобучает
лестницу моделей. Исключение — одна логистическая регрессия, которая нужна,
чтобы построить карту xG по полю для контролируемого сценария; она обучается
на том же train из сохранённого разбиения.

Запуск::

    python scripts/make_report.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xg_context.config import (
    FEATURE_SETS,
    FIGURES_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    REPORTS_DIR,
    TABLES_DIR,
    load_data_config,
)
from xg_context.data import build_downloader
from xg_context.dataset import _parse_freeze_frame
from xg_context.evaluation import evaluate_predictions
from xg_context.models import build_model, predict_proba
from xg_context.preprocessing import select_features
from xg_context.reporting import render_results_report
from xg_context.splitting import load_split
from xg_context.visualization import (
    apply_style,
    plot_ablation,
    plot_calibration,
    plot_error_analysis,
    plot_feature_importance,
    plot_freeze_frame_examples,
    plot_goal_rate_by_geometry,
    plot_sample_overview,
    plot_shot_map,
    plot_xg_surface,
    save_figure,
)

logger = logging.getLogger("make_report")

SHOTS_PATH = PROCESSED_DATA_DIR / "context_eligible_shots.parquet"
PREDICTIONS_PATH = PROCESSED_DATA_DIR / "test_predictions.parquet"
SPLIT_PATH = PROCESSED_DATA_DIR / "split.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument(
        "--skip-freeze-frames",
        action="store_true",
        help="не читать сырые события для примеров freeze frame",
    )
    return parser.parse_args(argv)


# --------------------------------------------------------------------------------------
# Анализ ошибок
# --------------------------------------------------------------------------------------


def build_subgroups(predictions: pd.DataFrame) -> pd.DataFrame:
    """Сравнить качество моделей без и с защитным контекстом по подгруппам ударов.

    Подгруппы выбраны по спецификации (раздел 13): удары головой против ударов
    ногой, центр против острых углов, open play против стандартов, плотная
    оборона против открытых моментов, ближние против дальних.
    """
    frame = predictions
    angle_deg = np.degrees(frame["shot_angle"])

    definitions: list[tuple[str, pd.Series]] = [
        ("Удары головой", frame["body_part"] == "Head"),
        ("Удары ногой", frame["body_part"].isin(["Right Foot", "Left Foot"])),
        ("Острый угол (< 15°)", angle_deg < 15),
        ("Центральная позиция (> 30°)", angle_deg > 30),
        ("Ближние удары (< 12 ярдов)", frame["shot_distance"] < 12),
        ("Средняя дистанция (12–20)", frame["shot_distance"].between(12, 20)),
        ("Дальние удары (> 20 ярдов)", frame["shot_distance"] > 20),
        ("Open play", frame["shot_type"] == "Open Play"),
        ("Стандарты", frame["shot_type"] != "Open Play"),
        ("Открытый момент (0 в конусе)", frame["opponents_in_shot_cone"] == 0),
        ("Плотная оборона (2+ в конусе)", frame["opponents_in_shot_cone"] >= 2),
        ("Под давлением", frame["under_pressure"].astype(bool)),
        ("Без давления", ~frame["under_pressure"].astype(bool)),
        ("Ближайший соперник < 2 ярдов", frame["nearest_opponent_distance"] < 2),
        ("Ближайший соперник > 5 ярдов", frame["nearest_opponent_distance"] > 5),
        ("Вратарь вышел (> 3 ярдов)", frame["goalkeeper_distance_to_goal_line"] > 3),
    ]

    rows: list[dict[str, Any]] = []
    for name, mask in definitions:
        part = frame[mask.fillna(False)]
        if len(part) < 50:
            continue
        without = evaluate_predictions(part["is_goal"], part["p_logistic_no_defense"])
        with_context = evaluate_predictions(part["is_goal"], part["p_logistic_defense"])
        statsbomb = evaluate_predictions(part["is_goal"], part["statsbomb_xg"])
        rows.append(
            {
                "подгруппа": name,
                "n": len(part),
                "доля_голов": float(part["is_goal"].mean()),
                "log_loss_без_контекста": without["log_loss"],
                "log_loss_с_контекстом": with_context["log_loss"],
                "log_loss_statsbomb": statsbomb["log_loss"],
                "улучшение": without["log_loss"] - with_context["log_loss"],
                "brier_без_контекста": without["brier"],
                "brier_с_контекстом": with_context["brier"],
            }
        )
    return pd.DataFrame(rows)


def pick_examples(predictions: pd.DataFrame) -> pd.DataFrame:
    """Выбрать удары, на которых защитный контекст меняет прогноз сильнее всего."""
    frame = predictions.copy()
    frame["Δ_контекст"] = frame["p_logistic_defense"] - frame["p_logistic_no_defense"]
    frame["Δ_statsbomb"] = frame["p_logistic_defense"] - frame["statsbomb_xg"]

    picks = pd.concat(
        [
            frame.nlargest(3, "Δ_контекст").assign(причина="контекст сильно повысил прогноз"),
            frame.nsmallest(3, "Δ_контекст").assign(причина="контекст сильно понизил прогноз"),
            frame.nlargest(3, "Δ_statsbomb").assign(причина="мы выше statsbomb_xg"),
            frame.nsmallest(3, "Δ_statsbomb").assign(причина="мы ниже statsbomb_xg"),
        ]
    )
    columns = [
        "причина",
        "shot_id",
        "match_id",
        "competition_name",
        "is_goal",
        "shot_distance",
        "shot_angle",
        "body_part",
        "shot_type",
        "n_opponents_visible",
        "opponents_in_shot_cone",
        "nearest_opponent_distance",
        "goalkeeper_distance_to_goal_line",
        "p_logistic_no_defense",
        "p_logistic_defense",
        "statsbomb_xg",
        "Δ_контекст",
        "Δ_statsbomb",
    ]
    return picks[columns].reset_index(drop=True)


def load_freeze_frames(config_path: str, shot_ids: list[str], match_ids: list[int]) -> dict:
    """Достать координаты freeze frame для выбранных ударов из сырых событий."""
    config = load_data_config(config_path)
    downloader = build_downloader(config)
    wanted = set(shot_ids)
    frames: dict[str, dict[str, Any]] = {}
    for match_id in sorted(set(match_ids)):
        for event in downloader.load_events(int(match_id)):
            if event.get("id") in wanted:
                shot = event.get("shot") or {}
                parsed = _parse_freeze_frame(shot.get("freeze_frame"))
                parsed["shot_x"] = event["location"][0]
                parsed["shot_y"] = event["location"][1]
                frames[event["id"]] = parsed
    return frames


# --------------------------------------------------------------------------------------
# Основной сценарий
# --------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args(argv)
    apply_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    shots = pd.read_parquet(SHOTS_PATH)
    predictions = pd.read_parquet(PREDICTIONS_PATH)
    ablation = pd.read_csv(TABLES_DIR / "ablation.csv")
    by_league = pd.read_csv(TABLES_DIR / "dataset_by_league.csv")
    summary = json.loads((TABLES_DIR / "experiment_summary.json").read_text(encoding="utf-8"))
    logger.info("Тестовых ударов: %d", len(predictions))

    # ------------------------------------------------------------------ данные
    logger.info("Графики: обзор данных")
    save_figure(plot_sample_overview(by_league), FIGURES_DIR / "01_sample_overview.png")
    save_figure(plot_shot_map(shots), FIGURES_DIR / "02_shot_map.png")
    save_figure(plot_goal_rate_by_geometry(shots), FIGURES_DIR / "03_goal_rate_by_geometry.png")

    # ------------------------------------------------------------------ ablation
    logger.info("Графики: ablation")
    save_figure(plot_ablation(ablation), FIGURES_DIR / "04_ablation.png")

    # ------------------------------------------------------------------ калибровка
    logger.info("Графики: калибровка")
    best_key = summary["best_model_key"]
    curves = {
        "M1: геометрия": "logistic_geometry",
        "M2: + характеристики и флаги": "logistic_geometry_shot_flags",
        "M4: + защитный контекст": "logistic_geometry_shot_flags_defensive",
        "Лучшая нелинейная + контекст": best_key.replace("@", "_"),
        "statsbomb_xg": "statsbomb_xg",
    }
    tables = {}
    for label, key in curves.items():
        path = TABLES_DIR / f"calibration_{key}.csv"
        if path.exists():
            tables[label] = pd.read_csv(path)
    save_figure(plot_calibration(tables), FIGURES_DIR / "05_calibration.png")

    # ------------------------------------------------------------------ важность
    logger.info("Графики: важность признаков")
    importance_path = TABLES_DIR / "feature_importance_logistic_geometry_shot_flags_defensive.csv"
    if importance_path.exists():
        importance = pd.read_csv(importance_path)
        importance["признак"] = importance["признак"].str.replace("num__", "", regex=False)
        importance["признак"] = importance["признак"].str.replace("cat__", "", regex=False)
        save_figure(plot_feature_importance(importance), FIGURES_DIR / "06_feature_importance.png")

    # ------------------------------------------------------------------ карта xG
    logger.info("Графики: карта xG для контролируемого сценария")
    split = load_split(SPLIT_PATH)
    train_matches = set(split["match_ids"]["train"])
    train = shots[shots["match_id"].isin(train_matches)]
    features = list(FEATURE_SETS["geometry_shot_flags_defensive"])
    model = build_model("logistic", features, params={"C": 1.0}, seed=RANDOM_SEED)
    model.fit(select_features(train, features), train["is_goal"].to_numpy())

    template = train[features].median(numeric_only=True)
    for column in features:
        if column not in template.index:
            template[column] = train[column].mode().iloc[0]
    template["body_part"] = "Right Foot"
    template["shot_technique"] = "Normal"
    template["shot_type"] = "Open Play"
    template["play_pattern"] = "Regular Play"

    def predict(grid: pd.DataFrame) -> np.ndarray:
        return predict_proba(model, select_features(grid, features))

    scenarios = [
        (
            "Открытый момент: никого в конусе,\nближайший соперник в 6 ярдах",
            {
                "opponents_in_shot_cone": 0,
                "opponents_between_shot_and_goal": 1,
                "nearest_opponent_distance": 6.0,
                "opponents_within_1y": 0,
                "opponents_within_2y": 0,
                "opponents_within_3y": 0,
                "opponents_within_5y": 0,
            },
        ),
        (
            "Плотная оборона: трое в конусе,\nближайший соперник в 1 ярде",
            {
                "opponents_in_shot_cone": 3,
                "opponents_between_shot_and_goal": 5,
                "nearest_opponent_distance": 1.0,
                "opponents_within_1y": 1,
                "opponents_within_2y": 2,
                "opponents_within_3y": 3,
                "opponents_within_5y": 4,
            },
        ),
    ]
    save_figure(
        plot_xg_surface(predict, template, scenarios=scenarios),
        FIGURES_DIR / "07_xg_surface.png",
    )

    # ------------------------------------------------------------------ ошибки
    logger.info("Анализ ошибок по подгруппам")
    subgroups = build_subgroups(predictions)
    subgroups.to_csv(TABLES_DIR / "error_analysis.csv", index=False, encoding="utf-8")
    save_figure(plot_error_analysis(subgroups), FIGURES_DIR / "08_error_analysis.png")
    logger.info(
        "Подгруппы:\n%s",
        subgroups[
            ["подгруппа", "n", "log_loss_без_контекста", "log_loss_с_контекстом", "улучшение"]
        ].to_string(index=False),
    )

    # ------------------------------------------------------------------ примеры
    examples = pick_examples(predictions)
    examples.to_csv(TABLES_DIR / "example_shots.csv", index=False, encoding="utf-8")

    if not args.skip_freeze_frames:
        logger.info("Графики: примеры freeze frame")
        # По одному удару на каждую причину: иначе все примеры окажутся
        # из одной категории и рисунок ничего не сравнивает.
        picked = (
            examples.drop_duplicates("shot_id")
            .groupby("причина", sort=False)
            .head(1)
            .head(4)
            .reset_index(drop=True)
        )
        frames = load_freeze_frames(
            args.config, picked["shot_id"].tolist(), picked["match_id"].tolist()
        )
        payload = []
        for row in picked.itertuples(index=False):
            frame = frames.get(row.shot_id)
            if frame is None:
                continue
            payload.append(
                {
                    **frame,
                    "title": f"{row.competition_name}: {row.причина}",
                    "caption": (
                        f"расстояние {row.shot_distance:.1f} ярда\n"
                        f"в конусе: {int(row.opponents_in_shot_cone)}\n"
                        f"ближайший: {row.nearest_opponent_distance:.1f} ярда\n"
                        f"без контекста {row.p_logistic_no_defense:.3f} → "
                        f"с контекстом {row.p_logistic_defense:.3f}\n"
                        f"statsbomb_xg {row.statsbomb_xg:.3f}, "
                        f"итог: {'гол' if row.is_goal else 'не гол'}"
                    ),
                }
            )
        if payload:
            save_figure(
                plot_freeze_frame_examples(payload), FIGURES_DIR / "09_freeze_frame_examples.png"
            )

    logger.info("Готово. Графики в %s", FIGURES_DIR)
    _write_results_report(summary, ablation, subgroups, examples)
    return 0


def _write_results_report(
    summary: dict[str, Any],
    ablation: pd.DataFrame,
    subgroups: pd.DataFrame,
    examples: pd.DataFrame,
) -> None:
    """Собрать `reports/results.md` из посчитанных таблиц."""
    text = render_results_report(
        summary,
        ablation,
        subgroups,
        examples,
        league=pd.read_csv(TABLES_DIR / "metrics_by_league.csv"),
        calibration=pd.read_csv(TABLES_DIR / "calibration_summary.csv"),
    )
    path = REPORTS_DIR / "results.md"
    path.write_text(text, encoding="utf-8")
    logger.info("Отчёт записан: %s", path)


if __name__ == "__main__":
    raise SystemExit(main())
