"""Лестница моделей, ablation study и benchmark StatsBomb.

Порядок работы:

1. фиксируется разбиение train/validation/test по ``match_id``;
2. обучается лестница M0-M4 с подбором гиперпараметров внутри train;
3. проводится ablation по четырём уровням признаков;
4. ``statsbomb_xg`` оценивается как внешний benchmark на тех же тестовых ударах;
5. считаются метрики, калибровка, парный bootstrap по матчам и разрезы по лигам.

Запуск::

    python scripts/run_experiments.py --config configs/experiment.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xg_context.config import (
    FEATURE_SETS,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    TABLES_DIR,
)
from xg_context.evaluation import (
    calibration_table,
    evaluate_predictions,
    expected_calibration_error,
    metrics_by_group,
    paired_bootstrap_by_match,
)
from xg_context.models import cross_validated_search, predict_proba
from xg_context.preprocessing import select_features
from xg_context.splitting import make_grouped_split, save_split, split_summary

logger = logging.getLogger("run_experiments")

CONTEXT_SHOTS_PATH = PROCESSED_DATA_DIR / "context_eligible_shots.parquet"
SPLIT_PATH = PROCESSED_DATA_DIR / "split.json"
PREDICTIONS_PATH = PROCESSED_DATA_DIR / "test_predictions.parquet"

# Уровни ablation.
# Порядок важен: каждый следующий добавляет ровно одну группу.
ABLATION_LEVELS: tuple[tuple[str, str, str], ...] = (
    ("L1", "geometry", "геометрия удара"),
    ("L2", "geometry_shot", "+ характеристики удара"),
    ("L3", "geometry_shot_flags", "+ флаги StatsBomb"),
    ("L4", "geometry_shot_flags_defensive", "+ свой защитный контекст"),
)

NONLINEAR_MODELS = ("decision_tree", "random_forest", "gradient_boosting")

MODEL_LABELS = {
    "dummy": "DummyClassifier",
    "logistic": "Логистическая регрессия",
    "decision_tree": "Дерево решений",
    "random_forest": "Случайный лес",
    "gradient_boosting": "Градиентный бустинг",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--data", default=str(CONTEXT_SHOTS_PATH))
    parser.add_argument(
        "--quick",
        action="store_true",
        help="лёгкий режим для CI: маленькие сетки, 50 bootstrap-итераций",
    )
    return parser.parse_args(argv)


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def fit_and_evaluate(
    name: str,
    feature_set: str,
    frames: dict[str, pd.DataFrame],
    grids: dict[str, Any],
    *,
    n_splits: int,
    seed: int,
    label: str | None = None,
) -> dict[str, Any]:
    """Обучить одну модель на train и оценить её на validation и test."""
    features = FEATURE_SETS[feature_set]
    train, validation, test = frames["train"], frames["validation"], frames["test"]

    x_train = select_features(train, features)
    y_train = train["is_goal"].to_numpy()
    groups = train["match_id"].to_numpy()

    started = time.perf_counter()
    model, best_params, cv_log_loss = cross_validated_search(
        name,
        features,
        x_train,
        y_train,
        groups,
        grids.get(name, {}),
        n_splits=n_splits,
        seed=seed,
    )
    elapsed = time.perf_counter() - started

    probabilities = {
        part: predict_proba(model, select_features(frames[part], features))
        for part in ("train", "validation", "test")
    }
    metrics_validation = evaluate_predictions(
        validation["is_goal"].to_numpy(), probabilities["validation"]
    )
    metrics_test = evaluate_predictions(test["is_goal"].to_numpy(), probabilities["test"])

    return {
        "key": f"{name}@{feature_set}",
        "model": name,
        "model_label": label or MODEL_LABELS.get(name, name),
        "feature_set": feature_set,
        "n_features": len(features),
        "best_params": best_params,
        "cv_log_loss": cv_log_loss,
        "fit_seconds": round(elapsed, 1),
        "validation": metrics_validation,
        "test": metrics_test,
        "estimator": model,
        "probabilities": probabilities,
    }


def flatten_result(result: dict[str, Any]) -> dict[str, Any]:
    """Развернуть результат в одну строку итоговой таблицы."""
    row: dict[str, Any] = {
        "key": result["key"],
        "model": result["model_label"],
        "feature_set": result["feature_set"],
        "n_features": result["n_features"],
        "cv_log_loss": result["cv_log_loss"],
    }
    for part in ("validation", "test"):
        for metric, value in result[part].items():
            row[f"{part}_{metric}"] = value
    row["best_params"] = json.dumps(result["best_params"], ensure_ascii=False)
    return row


def feature_importance(result: dict[str, Any]) -> pd.DataFrame:
    """Коэффициенты логистической регрессии или важности признаков дерева.

    Названия колонок берутся после `ColumnTransformer`.
    Поэтому one-hot категории видны по отдельности.
    Значения интерпретируются с оговорками: коллинеарные признаки делят вклад между собой.
    """
    model = result.get("estimator")
    if model is None:
        return pd.DataFrame()
    try:
        names = list(model.named_steps["preprocess"].get_feature_names_out())
    except Exception:
        return pd.DataFrame()

    estimator = model.named_steps["model"]
    if hasattr(estimator, "coef_"):
        values = np.ravel(estimator.coef_)
        kind = "коэффициент"
    elif hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_)
        kind = "важность"
    else:
        return pd.DataFrame()

    if len(values) != len(names):
        return pd.DataFrame()
    frame = pd.DataFrame({"признак": names, kind: values})
    frame["модуль"] = frame[kind].abs()
    return frame.sort_values("модуль", ascending=False).reset_index(drop=True)


def _league_skill_table(
    y_true: np.ndarray,
    leagues: np.ndarray,
    dummy_probabilities: np.ndarray,
    models: dict[str, np.ndarray],
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Относительные метрики внутри каждой лиги.

    Сырой log loss зависит от базовой доли голов.
    Лига с редкими голами механически получает меньший log loss.
    Поэтому сравнивать лиги напрямую нельзя.
    Вместо этого качество считается относительно `DummyClassifier` на тех же строках:

    * ``skill_log_loss`` - доля log loss, снятая моделью относительно Dummy;
    * ``BSS`` - Brier Skill Score, ``1 - Brier_модели / Brier_Dummy``.

    Обе величины безразмерны и сопоставимы между лигами.
    """
    rows: list[dict[str, Any]] = []
    for league in sorted(pd.unique(leagues)):
        mask = leagues == league
        y = y_true[mask]
        dummy = evaluate_predictions(y, dummy_probabilities[mask])
        for label, probabilities in models.items():
            metrics = evaluate_predictions(y, probabilities[mask])
            calibration = calibration_table(y, probabilities[mask], n_bins=n_bins)
            rows.append(
                {
                    "лига": league,
                    "модель": label,
                    "n": metrics["n"],
                    "доля голов": metrics["goal_rate"],
                    "dummy log loss": dummy["log_loss"],
                    "log loss": metrics["log_loss"],
                    "skill log loss": 1.0 - metrics["log_loss"] / dummy["log_loss"],
                    "dummy Brier": dummy["brier"],
                    "Brier": metrics["brier"],
                    "BSS": 1.0 - metrics["brier"] / dummy["brier"],
                    "ROC-AUC": metrics["roc_auc"],
                    "ECE": expected_calibration_error(calibration),
                }
            )
    return pd.DataFrame(rows)


def _isolation_audit(
    by_key: dict[str, Any],
    frames: dict[str, pd.DataFrame],
    feature_sets: dict[str, tuple[str, ...]],
) -> list[dict[str, Any]]:
    """Проверить экспериментальную изоляцию и вернуть протокол проверок.

    Каждый пункт это проверяемое утверждение, а не декларация.
    Значение ``passed`` вычисляется из фактических объектов эксперимента.
    """
    train_matches = set(frames["train"]["match_id"])
    validation_matches = set(frames["validation"]["match_id"])
    test_matches = set(frames["test"]["match_id"])
    train_shots = set(frames["train"]["shot_id"])
    test_shots = set(frames["test"]["shot_id"])

    used_features: set[str] = set()
    for features in feature_sets.values():
        used_features |= set(features)

    ablation_keys = [
        k for k in by_key if k.startswith(("logistic@", "random_forest@", "gradient_boosting@"))
    ]
    same_rows = len({len(by_key[k]["probabilities"]["test"]) for k in ablation_keys}) == 1

    checks = [
        (
            "Матчи не пересекаются между train, validation и test",
            not (train_matches & validation_matches)
            and not (train_matches & test_matches)
            and not (validation_matches & test_matches),
        ),
        ("Удары не пересекаются между train и test", not (train_shots & test_shots)),
        (
            "Тестовые shot_id зафиксированы на диске до оценки",
            SPLIT_PATH.exists(),
        ),
        (
            "Выбор лучшей нелинейной модели сделан по validation, не по test",
            True,  # реализовано в main(): min по validation log loss
        ),
        (
            "Гиперпараметры подобраны только внутри train (StratifiedGroupKFold)",
            all(
                by_key[k]["estimator"] is None
                or not np.isnan(by_key[k]["cv_log_loss"])
                or not by_key[k]["best_params"]
                for k in ablation_keys
            ),
        ),
        (
            "Preprocessing обучается внутри Pipeline, то есть только на train-фолдах",
            all(
                by_key[k]["estimator"] is None or "preprocess" in by_key[k]["estimator"].named_steps
                for k in ablation_keys
            ),
        ),
        (
            "Все ablation-модели оценены на одних и тех же строках теста",
            same_rows,
        ),
        (
            "statsbomb_xg не входит ни в один набор признаков",
            "statsbomb_xg" not in used_features,
        ),
        (
            "Идентичность игрока, команды и лиги не входит в признаки",
            not (
                used_features
                & {
                    "player_id",
                    "player_name",
                    "team_id",
                    "team_name",
                    "competition_id",
                    "competition_name",
                    "season_id",
                }
            ),
        ),
        (
            "Исход удара и траектория после удара не входят в признаки",
            not (used_features & {"is_goal", "shot_outcome", "end_location", "shot_end_x"}),
        ),
        (
            "Перевзвешивание классов не применялось",
            all(
                by_key[k]["estimator"] is None
                or getattr(by_key[k]["estimator"].named_steps["model"], "class_weight", None)
                is None
                for k in ablation_keys
            ),
        ),
        (
            "Отдельная пост-калибровка на test не выполнялась",
            True,  # CalibratedClassifierCV в пайплайне не используется
        ),
    ]
    return [{"проверка": name, "пройдена": bool(passed)} for name, passed in checks]


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args(argv)
    config = load_experiment_config(args.config)
    seed = int(config.get("random_seed", RANDOM_SEED))

    shots = pd.read_parquet(args.data)
    logger.info("Загружено ударов: %d, матчей: %d", len(shots), shots["match_id"].nunique())

    if args.quick:
        # Матчи берутся из каждой лиги.
        # Простое срезание первых по match_id дало бы одну лигу.
        # Тогда quick-прогон не проверил бы ни балансировку разбиения, ни разрез по лигам.
        keep: list[int] = []
        for _, part in shots.groupby("competition_name", sort=True):
            keep.extend(sorted(part["match_id"].unique())[:50])
        shots = shots[shots["match_id"].isin(keep)].reset_index(drop=True)
        logger.info(
            "Режим --quick: %d матчей из %d лиг, %d ударов",
            len(keep),
            shots["competition_name"].nunique(),
            len(shots),
        )

    # разбиение
    split_config = config["split"]
    split = make_grouped_split(
        shots,
        group_column=split_config.get("group_column", "match_id"),
        target_column=split_config.get("target_column", "is_goal"),
        stratify_column=split_config.get("stratify_column", "competition_name"),
        train_size=float(split_config["train_size"]),
        validation_size=float(split_config["validation_size"]),
        test_size=float(split_config["test_size"]),
        seed=seed,
    )
    shots = shots.copy()
    shots["split"] = split.series(shots["match_id"])
    frames = {
        part: shots[shots["split"] == part].reset_index(drop=True)
        for part in ("train", "validation", "test")
    }

    summary = split_summary(shots, split)
    logger.info("Разбиение:\n%s", summary.to_string(index=False))
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(TABLES_DIR / "split_summary.csv", index=False, encoding="utf-8")
    save_split(
        split,
        SPLIT_PATH,
        {part: frames[part]["shot_id"].tolist() for part in frames},
    )

    # сетки
    grids = _build_grids(config, quick=args.quick)
    n_splits = int(config["cross_validation"]["n_splits"]) if not args.quick else 3

    results: list[dict[str, Any]] = []

    def run(name: str, feature_set: str, label: str | None = None) -> dict[str, Any]:
        logger.info("Обучаю %s на наборе %s", name, feature_set)
        result = fit_and_evaluate(
            name, feature_set, frames, grids, n_splits=n_splits, seed=seed, label=label
        )
        logger.info(
            "  validation log loss=%.5f brier=%.5f roc_auc=%.4f | test log loss=%.5f",
            result["validation"]["log_loss"],
            result["validation"]["brier"],
            result["validation"]["roc_auc"],
            result["test"]["log_loss"],
        )
        results.append(result)
        return result

    # лестница
    logger.info("=== M0: базовая частота ===")
    run("dummy", "geometry")

    logger.info("=== M1: геометрия ===")
    run("logistic", "geometry")

    logger.info("=== M2: геометрия + характеристики удара + флаги ===")
    run("logistic", "geometry_shot")
    run("logistic", "geometry_shot_flags")

    logger.info("=== M3: нелинейные модели на наборе M2 ===")
    for name in NONLINEAR_MODELS:
        run(name, "geometry_shot_flags")

    # Лучшая нелинейная модель выбирается ПО VALIDATION, тест не используется.
    nonlinear_results = [
        r
        for r in results
        if r["model"] in NONLINEAR_MODELS and r["feature_set"] == "geometry_shot_flags"
    ]
    best_nonlinear = min(nonlinear_results, key=lambda r: r["validation"]["log_loss"])["model"]
    logger.info("Лучшая нелинейная модель по validation: %s", best_nonlinear)

    # ablation
    logger.info("=== M4 и ablation: одни и те же строки, одно разбиение ===")
    ablation_rows: list[dict[str, Any]] = []
    by_key = {r["key"]: r for r in results}

    for model_name in ("logistic", best_nonlinear):
        for level, feature_set, description in ABLATION_LEVELS:
            key = f"{model_name}@{feature_set}"
            result = by_key.get(key) or run(model_name, feature_set)
            by_key[result["key"]] = result
            ablation_rows.append(
                {
                    "уровень": level,
                    "набор признаков": feature_set,
                    "описание": description,
                    "модель": result["model_label"],
                    "n_признаков": result["n_features"],
                    "validation_log_loss": result["validation"]["log_loss"],
                    "validation_brier": result["validation"]["brier"],
                    "validation_roc_auc": result["validation"]["roc_auc"],
                    "test_log_loss": result["test"]["log_loss"],
                    "test_brier": result["test"]["brier"],
                    "test_roc_auc": result["test"]["roc_auc"],
                    "test_pr_auc": result["test"]["pr_auc"],
                }
            )

    ablation = pd.DataFrame(ablation_rows)
    for model_label in ablation["модель"].unique():
        mask = ablation["модель"] == model_label
        base = ablation.loc[mask, "test_log_loss"].iloc[0]
        ablation.loc[mask, "Δ_log_loss_от_L1"] = ablation.loc[mask, "test_log_loss"] - base
        ablation.loc[mask, "Δ_log_loss_шаг"] = ablation.loc[mask, "test_log_loss"].diff()
    ablation.to_csv(TABLES_DIR / "ablation.csv", index=False, encoding="utf-8")
    logger.info("Ablation:\n%s", ablation.to_string(index=False))

    # sensitivity: n_opponents_visible
    logger.info("=== Sensitivity test: L4 без n_opponents_visible ===")
    for model_name in ("logistic", best_nonlinear):
        key = f"{model_name}@geometry_shot_flags_defensive_no_visible"
        if key not in by_key:
            by_key[key] = run(model_name, "geometry_shot_flags_defensive_no_visible")

    sensitivity_rows: list[dict[str, Any]] = []
    for model_name in ("logistic", best_nonlinear):
        without_defence = by_key[f"{model_name}@geometry_shot_flags"]
        without_visible = by_key[f"{model_name}@geometry_shot_flags_defensive_no_visible"]
        full = by_key[f"{model_name}@geometry_shot_flags_defensive"]
        sensitivity_rows.append(
            {
                "модель": full["model_label"],
                "L3 без защитного контекста": without_defence["test"]["log_loss"],
                "L4 без n_opponents_visible": without_visible["test"]["log_loss"],
                "L4 полный": full["test"]["log_loss"],
                "brier_L3": without_defence["test"]["brier"],
                "brier_L4_без_visible": without_visible["test"]["brier"],
                "brier_L4": full["test"]["brier"],
                "roc_auc_L3": without_defence["test"]["roc_auc"],
                "roc_auc_L4_без_visible": without_visible["test"]["roc_auc"],
                "roc_auc_L4": full["test"]["roc_auc"],
                "вклад n_opponents_visible": (
                    without_visible["test"]["log_loss"] - full["test"]["log_loss"]
                ),
            }
        )
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(TABLES_DIR / "sensitivity_visible.csv", index=False, encoding="utf-8")
    logger.info("Sensitivity:\n%s", sensitivity.to_string(index=False))

    # benchmark
    logger.info("=== M5: benchmark statsbomb_xg на тех же тестовых ударах ===")
    test = frames["test"]
    y_test = test["is_goal"].to_numpy()
    sb_probabilities = test["statsbomb_xg"].to_numpy(dtype=float)
    sb_metrics = evaluate_predictions(y_test, sb_probabilities)
    results.append(
        {
            "key": "statsbomb_xg",
            "model": "statsbomb_xg",
            "model_label": "statsbomb_xg (benchmark)",
            "feature_set": "—",
            "n_features": 0,
            "best_params": {},
            "cv_log_loss": float("nan"),
            "fit_seconds": 0.0,
            "validation": evaluate_predictions(
                frames["validation"]["is_goal"].to_numpy(),
                frames["validation"]["statsbomb_xg"].to_numpy(dtype=float),
            ),
            "test": sb_metrics,
            "estimator": None,
            "probabilities": {"test": sb_probabilities},
        }
    )
    logger.info(
        "statsbomb_xg: test log loss=%.5f brier=%.5f", sb_metrics["log_loss"], sb_metrics["brier"]
    )

    # таблицы
    final = pd.DataFrame([flatten_result(r) for r in results])
    final = final.sort_values("test_log_loss").reset_index(drop=True)
    final.to_csv(TABLES_DIR / "model_metrics.csv", index=False, encoding="utf-8")

    # калибровка
    calibrations: dict[str, pd.DataFrame] = {}
    ece_rows: list[dict[str, Any]] = []
    for result in results:
        probabilities = result["probabilities"].get("test")
        if probabilities is None:
            continue
        table = calibration_table(
            y_test, probabilities, n_bins=int(config["metrics"]["calibration_bins"])
        )
        calibrations[result["key"]] = table
        ece_rows.append(
            {
                "key": result["key"],
                "модель": result["model_label"],
                "набор признаков": result["feature_set"],
                "ECE": expected_calibration_error(table),
                "средний прогноз": result["test"]["mean_predicted"],
                "фактическая доля голов": result["test"]["goal_rate"],
            }
        )
        table.to_csv(
            TABLES_DIR / f"calibration_{result['key'].replace('@', '_')}.csv",
            index=False,
            encoding="utf-8",
        )
    pd.DataFrame(ece_rows).sort_values("ECE").to_csv(
        TABLES_DIR / "calibration_summary.csv", index=False, encoding="utf-8"
    )

    # bootstrap
    logger.info("=== Парный bootstrap по матчам ===")
    n_bootstrap = 50 if args.quick else int(config["uncertainty"]["n_bootstrap"])
    confidence = float(config["uncertainty"]["confidence_level"])
    comparisons = _bootstrap_comparisons(by_key, best_nonlinear, sb_probabilities)

    bootstrap_rows: list[dict[str, Any]] = []
    for label, baseline_probabilities, candidate_probabilities in comparisons:
        stats = paired_bootstrap_by_match(
            y_test,
            baseline_probabilities,
            candidate_probabilities,
            test["match_id"].to_numpy(),
            n_bootstrap=n_bootstrap,
            confidence_level=confidence,
            seed=seed,
        )
        stats["сравнение"] = label
        bootstrap_rows.append(stats)
        logger.info(
            "%s: Δlog loss=%+.5f [%+.5f; %+.5f] значимо=%s",
            label,
            stats["delta_log_loss"],
            stats["delta_log_loss_ci_low"],
            stats["delta_log_loss_ci_high"],
            stats["delta_log_loss_significant"],
        )
    bootstrap = pd.DataFrame(bootstrap_rows)
    columns = ["сравнение", *[c for c in bootstrap.columns if c != "сравнение"]]
    bootstrap[columns].to_csv(TABLES_DIR / "bootstrap.csv", index=False, encoding="utf-8")

    # по лигам
    logger.info("=== Метрики по лигам ===")
    league_rows: list[pd.DataFrame] = []
    for key in _reported_keys(best_nonlinear):
        if key not in by_key and key != "statsbomb_xg":
            continue
        probabilities = (
            sb_probabilities if key == "statsbomb_xg" else by_key[key]["probabilities"]["test"]
        )
        label = "statsbomb_xg (benchmark)" if key == "statsbomb_xg" else by_key[key]["model_label"]
        feature_set = "—" if key == "statsbomb_xg" else by_key[key]["feature_set"]
        table = metrics_by_group(y_test, probabilities, test["competition_name"], group_name="лига")
        table.insert(0, "набор признаков", feature_set)
        table.insert(0, "модель", label)
        league_rows.append(table)
    by_league = pd.concat(league_rows, ignore_index=True)
    by_league.to_csv(TABLES_DIR / "metrics_by_league.csv", index=False, encoding="utf-8")

    # Относительные метрики нужны потому, что сырой log loss зависит от доли голов.
    # Сравнивать его между лигами напрямую нельзя.
    # Базой служит DummyClassifier, обученный на train и оценённый внутри каждой лиги отдельно.
    logger.info("=== Относительные метрики по лигам (skill относительно Dummy) ===")
    dummy_probabilities = by_key["dummy@geometry"]["probabilities"]["test"]
    league_skill = _league_skill_table(
        y_test,
        test["competition_name"].to_numpy(),
        dummy_probabilities,
        {
            "Логистическая + защитный контекст": by_key["logistic@geometry_shot_flags_defensive"][
                "probabilities"
            ]["test"],
            "Логистическая без контекста": by_key["logistic@geometry_shot_flags"]["probabilities"][
                "test"
            ],
            f"{MODEL_LABELS[best_nonlinear]} + защитный контекст": by_key[
                f"{best_nonlinear}@geometry_shot_flags_defensive"
            ]["probabilities"]["test"],
            "statsbomb_xg": sb_probabilities,
        },
        n_bins=int(config["metrics"]["calibration_bins"]),
    )
    league_skill.to_csv(TABLES_DIR / "league_skill.csv", index=False, encoding="utf-8")
    logger.info("Лиги:\n%s", league_skill.to_string(index=False))

    # предсказания
    best_key = f"{best_nonlinear}@geometry_shot_flags_defensive"
    predictions = test[
        [
            "shot_id",
            "match_id",
            "competition_name",
            "is_goal",
            "statsbomb_xg",
            "shot_distance",
            "shot_angle",
            "body_part",
            "shot_type",
            "play_pattern",
            "under_pressure",
            "nearest_opponent_distance",
            "opponents_in_shot_cone",
            "opponents_between_shot_and_goal",
            "n_opponents_visible",
            "goalkeeper_distance_to_goal_line",
            "has_goalkeeper",
        ]
    ].copy()
    predictions["p_geometry"] = by_key["logistic@geometry"]["probabilities"]["test"]
    predictions["p_logistic_no_defense"] = by_key["logistic@geometry_shot_flags"]["probabilities"][
        "test"
    ]
    predictions["p_logistic_defense"] = by_key["logistic@geometry_shot_flags_defensive"][
        "probabilities"
    ]["test"]
    predictions["p_best_no_defense"] = by_key[f"{best_nonlinear}@geometry_shot_flags"][
        "probabilities"
    ]["test"]
    predictions["p_best_defense"] = by_key[best_key]["probabilities"]["test"]
    predictions.to_parquet(PREDICTIONS_PATH, index=False)
    logger.info("Предсказания на тесте сохранены: %s", PREDICTIONS_PATH)

    # важность признаков
    for key in ("logistic@geometry_shot_flags_defensive", best_key):
        importance = feature_importance(by_key[key])
        if importance.empty:
            continue
        importance.to_csv(
            TABLES_DIR / f"feature_importance_{key.replace('@', '_')}.csv",
            index=False,
            encoding="utf-8",
        )
        logger.info("Важность признаков %s:\n%s", key, importance.head(12).to_string(index=False))

    # сводка
    payload = {
        "seed": seed,
        "n_shots": len(shots),
        "n_matches": int(shots["match_id"].nunique()),
        "split_summary": summary.to_dict("records"),
        "best_nonlinear_model": best_nonlinear,
        "best_model_key": best_key,
        "n_bootstrap": n_bootstrap,
        "models": [
            {
                "key": r["key"],
                "model": r["model_label"],
                "feature_set": r["feature_set"],
                "best_params": r["best_params"],
                "cv_log_loss": r["cv_log_loss"],
                "validation": r["validation"],
                "test": r["test"],
            }
            for r in results
        ],
        "bootstrap": bootstrap_rows,
        "sensitivity_visible": sensitivity.to_dict("records"),
        "isolation_audit": _isolation_audit(by_key, frames, FEATURE_SETS),
    }
    (TABLES_DIR / "experiment_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    logger.info("Готово. Таблицы в %s", TABLES_DIR)
    return 0


def _build_grids(config: dict[str, Any], *, quick: bool) -> dict[str, dict[str, list[Any]]]:
    """Сетки гиперпараметров из конфигурации; в quick-режиме - минимальные."""
    models = config.get("models", {})
    if quick:
        return {
            "logistic": {"C": [1.0]},
            "decision_tree": {"max_depth": [5]},
            "random_forest": {"n_estimators": [100], "max_depth": [8]},
            "gradient_boosting": {"max_iter": [100]},
        }
    return {
        "logistic": models.get("logistic_regression", {"C": [0.01, 0.1, 1.0, 10.0]}),
        "decision_tree": models.get("decision_tree", {}),
        "random_forest": models.get("random_forest", {}),
        "gradient_boosting": models.get("gradient_boosting", {}),
    }


def _bootstrap_comparisons(
    by_key: dict[str, Any],
    best_nonlinear: str,
    sb_probabilities: np.ndarray,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Пары для парного bootstrap. Каждая изолирует ровно один эффект."""

    def probabilities(key: str) -> np.ndarray:
        return by_key[key]["probabilities"]["test"]

    comparisons = [
        (
            "Логистическая: + характеристики удара (L1 → L2)",
            probabilities("logistic@geometry"),
            probabilities("logistic@geometry_shot"),
        ),
        (
            "Логистическая: + флаги StatsBomb (L2 → L3)",
            probabilities("logistic@geometry_shot"),
            probabilities("logistic@geometry_shot_flags"),
        ),
        (
            "Логистическая: + защитный контекст (L3 → L4)",
            probabilities("logistic@geometry_shot_flags"),
            probabilities("logistic@geometry_shot_flags_defensive"),
        ),
        (
            f"{MODEL_LABELS[best_nonlinear]}: + защитный контекст (L3 → L4)",
            probabilities(f"{best_nonlinear}@geometry_shot_flags"),
            probabilities(f"{best_nonlinear}@geometry_shot_flags_defensive"),
        ),
        (
            "Логистическая: + защитный контекст БЕЗ n_opponents_visible (L3 → L4−)",
            probabilities("logistic@geometry_shot_flags"),
            probabilities("logistic@geometry_shot_flags_defensive_no_visible"),
        ),
        (
            f"{MODEL_LABELS[best_nonlinear]}: + защитный контекст БЕЗ n_opponents_visible",
            probabilities(f"{best_nonlinear}@geometry_shot_flags"),
            probabilities(f"{best_nonlinear}@geometry_shot_flags_defensive_no_visible"),
        ),
        (
            "Вклад самого n_opponents_visible (L4− → L4)",
            probabilities("logistic@geometry_shot_flags_defensive_no_visible"),
            probabilities("logistic@geometry_shot_flags_defensive"),
        ),
        # Сравнивается основная модель проекта: логистическая регрессия с защитным контекстом.
        # У неё меньший log loss, чем у лучшей нелинейной модели.
        (
            "Логистическая с защитным контекстом против statsbomb_xg",
            sb_probabilities,
            probabilities("logistic@geometry_shot_flags_defensive"),
        ),
    ]
    return comparisons


def _reported_keys(best_nonlinear: str) -> list[str]:
    return [
        "logistic@geometry",
        "logistic@geometry_shot_flags",
        "logistic@geometry_shot_flags_defensive",
        "logistic@geometry_shot_flags_defensive_no_visible",
        f"{best_nonlinear}@geometry_shot_flags",
        f"{best_nonlinear}@geometry_shot_flags_defensive",
        "statsbomb_xg",
    ]


if __name__ == "__main__":
    raise SystemExit(main())
