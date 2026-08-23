"""Smoke-тест полного пайплайна на синтетических событиях (спецификация, раздел 17).

Проверяется, что путь «события StatsBomb → датасет → разбиение → модель →
метрики → bootstrap» проходит целиком и даёт ожидаемые таблицы. Сеть не нужна:
события генерируются в схеме StatsBomb, поэтому тест выполняется в CI
без скачивания данных.

Это проверка связности пайплайна, а не качества модели: на синтетике
осмысленных метрик не бывает, поэтому утверждения касаются только структуры
результатов и соблюдения контрактов.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from xg_context.config import FEATURE_SETS, RANDOM_SEED
from xg_context.dataset import (
    FilterLog,
    SchemaReport,
    add_defensive_features,
    add_geometry_features,
    add_target,
    apply_shot_filters,
    extract_shot_rows,
    split_eligible_frames,
)
from xg_context.evaluation import (
    calibration_table,
    evaluate_predictions,
    expected_calibration_error,
    metrics_by_group,
    paired_bootstrap_by_match,
)
from xg_context.models import build_model, predict_proba
from xg_context.preprocessing import select_features
from xg_context.splitting import make_grouped_split, split_summary

LEAGUES = ("La Liga", "Premier League", "Serie A", "Ligue 1")
OUTCOMES = ("Goal", "Saved", "Blocked", "Off T", "Wayward", "Post")
BODY_PARTS = ("Right Foot", "Left Foot", "Head")


def _shot_event(rng: np.random.Generator, index: int) -> dict[str, Any]:
    """Одно событие удара в схеме StatsBomb Open Data."""
    shot_x = float(rng.uniform(84.0, 119.0))
    shot_y = float(rng.uniform(18.0, 62.0))

    # Вероятность гола растёт с близостью — чтобы модель могла хоть что-то выучить.
    distance = float(np.hypot(120.0 - shot_x, 40.0 - shot_y))
    probability = float(np.clip(0.75 - 0.035 * distance, 0.02, 0.6))
    outcome = "Goal" if rng.random() < probability else str(rng.choice(OUTCOMES[1:]))

    n_opponents = int(rng.integers(3, 10))
    freeze_frame = [
        {
            "location": [float(rng.uniform(shot_x, 119.5)), float(rng.uniform(20.0, 60.0))],
            "teammate": False,
            "position": {"name": "Centre Back"},
            "player": {"id": 100 + i, "name": f"Защитник {i}"},
        }
        for i in range(n_opponents)
    ]
    freeze_frame.append(
        {
            "location": [float(rng.uniform(114.0, 119.5)), float(rng.uniform(36.0, 44.0))],
            "teammate": False,
            "position": {"name": "Goalkeeper"},
            "player": {"id": 1, "name": "Вратарь"},
        }
    )
    freeze_frame.append(
        {
            "location": [float(rng.uniform(80.0, 110.0)), float(rng.uniform(20.0, 60.0))],
            "teammate": True,
            "position": {"name": "Striker"},
            "player": {"id": 2, "name": "Партнёр"},
        }
    )

    shot: dict[str, Any] = {
        "outcome": {"id": 97, "name": outcome},
        "type": {"id": 87, "name": "Open Play"},
        "body_part": {"id": 40, "name": str(rng.choice(BODY_PARTS))},
        "technique": {"id": 93, "name": "Normal"},
        "statsbomb_xg": float(probability),
        "freeze_frame": freeze_frame,
    }
    if rng.random() < 0.25:
        shot["first_time"] = True
    if rng.random() < 0.10:
        shot["one_on_one"] = True

    event: dict[str, Any] = {
        "id": f"shot-{index:06d}",
        "type": {"id": 16, "name": "Shot"},
        "period": int(rng.integers(1, 3)),
        "minute": int(rng.integers(0, 90)),
        "second": 0,
        "location": [shot_x, shot_y],
        "play_pattern": {"id": 1, "name": "Regular Play"},
        "player": {"id": 10, "name": "Бьющий"},
        "team": {"id": 5, "name": "Команда"},
        "shot": shot,
    }
    if rng.random() < 0.3:
        event["under_pressure"] = True
    return event


@pytest.fixture(scope="module")
def dataset() -> tuple[pd.DataFrame, pd.DataFrame, FilterLog, SchemaReport]:
    """Построить датасет из синтетических матчей всем реальным пайплайном."""
    rng = np.random.default_rng(RANDOM_SEED)
    schema = SchemaReport()
    rows: list[dict[str, Any]] = []
    index = 0

    for league_id, league in enumerate(LEAGUES, start=1):
        for match in range(30):
            match_id = league_id * 1000 + match
            events: list[dict[str, Any]] = []
            for _ in range(int(rng.integers(18, 30))):
                index += 1
                events.append(_shot_event(rng, index))
            # Пенальти, который обязан быть отфильтрован.
            index += 1
            penalty = _shot_event(rng, index)
            penalty["shot"]["type"] = {"id": 88, "name": "Penalty"}
            events.append(penalty)
            # Не-удар: должен быть проигнорирован разбором.
            events.append({"id": f"pass-{index}", "type": {"id": 30, "name": "Pass"}})

            meta = {
                "match_id": match_id,
                "competition_id": league_id,
                "season_id": 27,
                "competition_name": league,
                "match_date": "2016-01-01",
            }
            rows.extend(extract_shot_rows(events, meta, schema))

    raw = pd.DataFrame(rows)
    log = FilterLog()
    shots = apply_shot_filters(raw, log)
    shots = add_target(shots)
    shots = add_geometry_features(shots)
    shots = add_defensive_features(shots)
    all_eligible, context_eligible = split_eligible_frames(shots)
    return all_eligible, context_eligible, log, schema


class TestDatasetStage:
    def test_penalties_are_filtered_with_counters(self, dataset) -> None:
        _, _, log, _ = dataset
        steps = {step["step"]: step for step in log.steps}
        assert steps["drop_penalties"]["n_dropped"] == 120, "по одному пенальти на матч"
        assert steps["drop_unknown_outcome"]["n_dropped"] == 0

    def test_every_filter_is_logged(self, dataset) -> None:
        _, _, log, _ = dataset
        frame = log.to_frame()
        assert set(frame.columns) >= {"step", "n_before", "n_after", "n_dropped"}
        assert (frame["n_after"] <= frame["n_before"]).all()

    def test_schema_report_sees_only_known_outcomes(self, dataset) -> None:
        _, _, _, schema = dataset
        assert schema.unknown_outcomes == {}

    def test_sparse_booleans_are_true_only(self, dataset) -> None:
        """Синтетика воспроизводит поведение StatsBomb: поле есть только когда true."""
        _, _, _, schema = dataset
        assert all(schema.sparse_booleans_confirmed.values())

    def test_target_matches_outcome(self, dataset) -> None:
        all_eligible, _, _, _ = dataset
        assert set(all_eligible["is_goal"].unique()) <= {0, 1}
        assert (all_eligible.loc[all_eligible["shot_outcome"] == "Goal", "is_goal"] == 1).all()

    def test_all_model_features_are_present(self, dataset) -> None:
        _, context, _, _ = dataset
        for name, features in FEATURE_SETS.items():
            missing = set(features) - set(context.columns)
            assert not missing, f"набор {name}: нет {sorted(missing)}"

    def test_goalkeeper_is_not_counted_among_field_opponents(self, dataset) -> None:
        _, context, _, _ = dataset
        # В каждом кадре 3..9 полевых и один вратарь.
        assert context["n_opponents_visible"].between(3, 9).all()
        assert context["has_goalkeeper"].all()


@pytest.fixture(scope="module")
def trained(dataset):
    """Обучить модель на train синтетического датасета и предсказать на test."""
    _, context, _, _ = dataset
    split = make_grouped_split(context, seed=RANDOM_SEED)
    frame = context.copy()
    frame["split"] = split.series(frame["match_id"])
    train = frame[frame["split"] == "train"]
    test = frame[frame["split"] == "test"]

    features = FEATURE_SETS["geometry_shot_flags_defensive"]
    model = build_model("logistic", features, seed=RANDOM_SEED)
    model.fit(select_features(train, features), train["is_goal"])
    probabilities = predict_proba(model, select_features(test, features))
    return frame, split, test, probabilities


class TestExperimentStage:
    def test_split_covers_every_shot_once(self, dataset, trained) -> None:
        _, context, _, _ = dataset
        frame, _, _, _ = trained
        assert frame["split"].notna().all()
        assert len(frame) == len(context)

    def test_split_is_balanced(self, dataset, trained) -> None:
        _, context, _, _ = dataset
        _, split, _, _ = trained
        summary = split_summary(context, split)
        shares = dict(zip(summary["split"], summary["share_of_shots"], strict=True))
        assert shares["train"] == pytest.approx(0.70, abs=0.03)
        assert summary["n_goals"].gt(0).all()

    def test_predictions_are_valid_probabilities(self, trained) -> None:
        _, _, _, probabilities = trained
        assert np.isfinite(probabilities).all()
        assert ((probabilities > 0) & (probabilities < 1)).all()

    def test_metrics_have_expected_shape(self, trained) -> None:
        _, _, test, probabilities = trained
        metrics = evaluate_predictions(test["is_goal"], probabilities)
        assert set(metrics) >= {"log_loss", "brier", "roc_auc", "pr_auc", "n", "n_goals"}
        assert metrics["log_loss"] > 0
        assert 0 <= metrics["brier"] <= 1

    def test_model_beats_the_base_rate(self, trained) -> None:
        """Связность пайплайна: обученная модель не хуже предсказания частоты."""
        frame, _, test, probabilities = trained
        train_rate = frame.loc[frame["split"] == "train", "is_goal"].mean()
        baseline = np.full(len(test), train_rate)
        assert (
            evaluate_predictions(test["is_goal"], probabilities)["log_loss"]
            < evaluate_predictions(test["is_goal"], baseline)["log_loss"]
        )

    def test_calibration_table_is_well_formed(self, trained) -> None:
        _, _, test, probabilities = trained
        table = calibration_table(test["is_goal"], probabilities, n_bins=5)
        assert not table.empty
        assert table["n"].sum() == len(test)
        assert np.isfinite(expected_calibration_error(table))

    def test_metrics_by_league_cover_all_leagues(self, trained) -> None:
        _, _, test, probabilities = trained
        table = metrics_by_group(test["is_goal"], probabilities, test["competition_name"])
        assert set(table["группа"]) == set(LEAGUES)

    def test_paired_bootstrap_returns_interval(self, trained) -> None:
        _, _, test, probabilities = trained
        baseline = np.full(len(test), float(test["is_goal"].mean()))
        stats = paired_bootstrap_by_match(
            test["is_goal"],
            baseline,
            probabilities,
            test["match_id"],
            n_bootstrap=50,
            seed=RANDOM_SEED,
        )
        assert stats["delta_log_loss_ci_low"] <= stats["delta_log_loss"]
        assert stats["delta_log_loss"] <= stats["delta_log_loss_ci_high"]
        assert stats["n_matches"] == test["match_id"].nunique()

    def test_benchmark_can_be_scored_on_the_same_rows(self, trained) -> None:
        """statsbomb_xg оценивается на тех же тестовых ударах, что и модель."""
        _, _, test, probabilities = trained
        benchmark = evaluate_predictions(test["is_goal"], test["statsbomb_xg"])
        model = evaluate_predictions(test["is_goal"], probabilities)
        assert benchmark["n"] == model["n"]
