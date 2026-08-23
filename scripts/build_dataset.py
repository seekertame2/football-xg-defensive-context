"""Построение shot-level датасета утверждённой выборки (этап 2 спецификации).

Скрипт один раз разбирает события всех матчей выборки и делает три вещи:

1. строит таблицы ``all_eligible_shots`` и ``context_eligible_shots``;
2. пишет журнал фильтрации, отчёт о схеме и проверку покрытия по полной выборке
   (эти числа затем использует `scripts/audit_data.py --selection`);
3. обновляет `data/data_manifest.json`.

Запуск::

    python scripts/build_dataset.py --config configs/data.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xg_context.config import (
    DATA_MANIFEST_PATH,
    DATASET_CONFIG_VERSION,
    PROCESSED_DATA_DIR,
    TABLES_DIR,
    load_data_config,
)
from xg_context.data import build_downloader, matches_path
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

logger = logging.getLogger("build_dataset")

ALL_SHOTS_PATH = PROCESSED_DATA_DIR / "all_eligible_shots.parquet"
CONTEXT_SHOTS_PATH = PROCESSED_DATA_DIR / "context_eligible_shots.parquet"
FULL_AUDIT_PATH = TABLES_DIR / "full_sample_audit.json"

#: Ниже этого покрытия защитного контекста исследовательский дизайн
#: пришлось бы пересматривать, и скрипт останавливается (требование владельца).
MIN_ACCEPTABLE_CONTEXT_SHARE = 0.95

#: Колонки со списками координат: нужны для расчёта признаков,
#: но в итоговых таблицах не хранятся.
FRAME_COLUMNS = ("opponent_x", "opponent_y")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument(
        "--limit-matches",
        type=int,
        default=None,
        help="ограничить число матчей (отладка и CI)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="лёгкий режим: 40 матчей (по 10 на лигу), вывод с суффиксом _quick",
    )
    return parser.parse_args(argv)


def load_selected_matches(downloader, selection: dict[str, Any]) -> list[dict[str, Any]]:
    """Собрать список матчей утверждённой выборки с метаданными соревнования."""
    matches: list[dict[str, Any]] = []
    for item in selection["competition_seasons"]:
        competition_id = int(item["competition_id"])
        season_id = int(item["season_id"])
        payload = downloader.load_json(matches_path(competition_id, season_id))
        for match in payload:
            competition = match.get("competition") or {}
            matches.append(
                {
                    "match_id": int(match["match_id"]),
                    "competition_id": competition_id,
                    "season_id": season_id,
                    "competition_name": competition.get("competition_name", str(competition_id)),
                    "match_date": match.get("match_date"),
                }
            )
    matches.sort(key=lambda m: (m["competition_id"], m["match_id"]))
    return matches


def build_full_sample_audit(
    all_eligible: pd.DataFrame,
    context_eligible: pd.DataFrame,
    raw_shots: pd.DataFrame,
    schema: SchemaReport,
    filter_log: FilterLog,
) -> dict[str, Any]:
    """Проверка покрытия по ПОЛНОЙ выборке — заменяет оценку по 3 матчам на сезон."""
    n_all = len(all_eligible)
    by_league = (
        all_eligible.groupby("competition_name")
        .agg(
            n_matches=("match_id", "nunique"),
            n_shots=("shot_id", "size"),
            goal_rate=("is_goal", "mean"),
            share_with_freeze_frame=("has_freeze_frame", "mean"),
            share_with_opponents=("has_opponent_coordinates", "mean"),
            share_with_goalkeeper=("has_goalkeeper", "mean"),
            share_with_statsbomb_xg=("statsbomb_xg", lambda s: float(s.notna().mean())),
            mean_opponents_visible=("n_opponents_visible", "mean"),
            median_opponents_visible=("n_opponents_visible", "median"),
        )
        .reset_index()
        .sort_values("n_shots", ascending=False)
    )

    return {
        "n_matches": int(all_eligible["match_id"].nunique()),
        "n_events_total": schema.n_events_total,
        "n_shot_events": len(raw_shots),
        "n_all_eligible": n_all,
        "n_context_eligible": len(context_eligible),
        "share_context_of_all": float(len(context_eligible) / n_all) if n_all else 0.0,
        "share_with_freeze_frame": float(all_eligible["has_freeze_frame"].mean()),
        "share_with_opponent_coordinates": float(all_eligible["has_opponent_coordinates"].mean()),
        "share_with_goalkeeper": float(all_eligible["has_goalkeeper"].mean()),
        "share_with_statsbomb_xg": float(all_eligible["statsbomb_xg"].notna().mean()),
        "goal_rate_all_eligible": float(all_eligible["is_goal"].mean()),
        "goal_rate_context_eligible": float(context_eligible["is_goal"].mean())
        if len(context_eligible)
        else float("nan"),
        "mean_opponents_visible": float(all_eligible["n_opponents_visible"].mean()),
        "median_opponents_visible": float(all_eligible["n_opponents_visible"].median()),
        "n_frames_with_invalid_locations": int(
            (all_eligible["n_frame_invalid_locations"] > 0).sum()
        ),
        "by_league": by_league.to_dict("records"),
        "schema": schema.to_dict(),
        "filter_log": filter_log.steps,
    }


def _spread_across_competitions(matches: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Взять ограниченное число матчей поровну из каждого соревнования.

    Простое срезание первых N дало бы матчи одной лиги: список отсортирован
    по competition_id. Тогда smoke-прогон не проверил бы ни балансировку
    разбиения по лигам, ни разрез метрик по ним.
    """
    by_competition: dict[int, list[dict[str, Any]]] = {}
    for match in matches:
        by_competition.setdefault(match["competition_id"], []).append(match)

    picked: list[dict[str, Any]] = []
    per_competition = max(1, limit // max(len(by_competition), 1))
    for group in by_competition.values():
        picked.extend(group[:per_competition])
    return sorted(picked, key=lambda m: (m["competition_id"], m["match_id"]))[:limit]


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args(argv)
    config = load_data_config(args.config)
    selection = config.selection or {}

    if not selection.get("approved", False):
        raise SystemExit(
            "Выборка в configs/data.yaml не утверждена (selection.approved = false).\n"
            "Сначала выполните аудит и утвердите состав выборки."
        )

    suffix = ""
    limit = args.limit_matches
    if args.quick:
        limit = 40
        suffix = "_quick"
        logger.info("Режим --quick: 20 матчей, вывод с суффиксом _quick.")

    downloader = build_downloader(config)
    matches = load_selected_matches(downloader, selection)
    if limit is not None:
        matches = _spread_across_competitions(matches, limit)
    logger.info("Матчей в выборке: %d", len(matches))

    logger.info("Шаг 1/5: разбор событий")
    schema = SchemaReport(n_matches=len(matches))
    rows: list[dict[str, Any]] = []
    for index, match in enumerate(matches, start=1):
        events = downloader.load_events(match["match_id"])
        rows.extend(extract_shot_rows(events, match, schema))
        if index % 250 == 0:
            logger.info("  разобрано матчей: %d / %d, ударов: %d", index, len(matches), len(rows))

    raw_shots = pd.DataFrame(rows)
    logger.info("Событий: %d, из них ударов: %d", schema.n_events_total, len(raw_shots))

    if schema.unknown_outcomes:
        logger.error("Неизвестные исходы удара: %s", schema.unknown_outcomes)

    logger.info("Шаг 2/5: фильтры")
    filter_log = FilterLog()
    shots = apply_shot_filters(raw_shots, filter_log)
    filter_log.log()

    logger.info("Шаг 3/5: target и признаки")
    shots = add_target(shots)
    shots = add_geometry_features(shots)
    shots = add_defensive_features(shots)

    all_eligible, context_eligible = split_eligible_frames(shots)
    filter_log.record(
        "context_eligible",
        "Есть freeze_frame и хотя бы один соперник с координатами",
        len(all_eligible),
        len(context_eligible),
    )

    logger.info("Шаг 4/5: проверка покрытия по полной выборке")
    audit = build_full_sample_audit(all_eligible, context_eligible, raw_shots, schema, filter_log)
    logger.info(
        "Покрытие: freeze_frame %.4f, координаты соперников %.4f, вратарь %.4f, statsbomb_xg %.4f",
        audit["share_with_freeze_frame"],
        audit["share_with_opponent_coordinates"],
        audit["share_with_goalkeeper"],
        audit["share_with_statsbomb_xg"],
    )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    (TABLES_DIR / f"full_sample_audit{suffix}.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    filter_log.to_frame().to_csv(
        TABLES_DIR / f"dataset_filters{suffix}.csv", index=False, encoding="utf-8"
    )
    pd.DataFrame(audit["by_league"]).to_csv(
        TABLES_DIR / f"dataset_by_league{suffix}.csv", index=False, encoding="utf-8"
    )

    # Останов по требованию владельца проекта, если данные оказались хуже ожидаемого.
    problems: list[str] = []
    if audit["share_with_freeze_frame"] < MIN_ACCEPTABLE_CONTEXT_SHARE:
        problems.append(
            f"покрытие freeze_frame {audit['share_with_freeze_frame']:.4f} "
            f"ниже порога {MIN_ACCEPTABLE_CONTEXT_SHARE}"
        )
    if schema.unknown_outcomes:
        problems.append(f"неизвестные исходы удара: {schema.unknown_outcomes}")
    if problems:
        raise SystemExit(
            "Остановка после проверки данных. Обнаружено:\n  - "
            + "\n  - ".join(problems)
            + f"\nПодробности: {TABLES_DIR / f'full_sample_audit{suffix}.json'}"
        )

    logger.info("Шаг 5/5: запись датасета и манифеста")
    store_all = all_eligible.drop(columns=list(FRAME_COLUMNS))
    store_context = context_eligible.drop(columns=list(FRAME_COLUMNS))
    all_path = PROCESSED_DATA_DIR / f"all_eligible_shots{suffix}.parquet"
    context_path = PROCESSED_DATA_DIR / f"context_eligible_shots{suffix}.parquet"
    store_all.to_parquet(all_path, index=False)
    store_context.to_parquet(context_path, index=False)
    logger.info("Записано: %s (%d строк)", all_path, len(store_all))
    logger.info("Записано: %s (%d строк)", context_path, len(store_context))

    if not args.quick:
        _update_manifest(config, audit, matches, all_path, context_path)

    logger.info(
        "Готово. all_eligible=%d, context_eligible=%d, доля голов=%.4f",
        len(all_eligible),
        len(context_eligible),
        audit["goal_rate_all_eligible"],
    )
    return 0


def _update_manifest(config, audit: dict[str, Any], matches, all_path, context_path) -> None:
    """Дописать в манифест всё, что требует раздел 5.4 спецификации."""
    import hashlib

    manifest: dict[str, Any] = {}
    if DATA_MANIFEST_PATH.exists():
        manifest = json.loads(DATA_MANIFEST_PATH.read_text(encoding="utf-8"))

    match_ids = sorted(m["match_id"] for m in matches)
    match_ids_hash = hashlib.sha256(",".join(str(m) for m in match_ids).encode("utf-8")).hexdigest()

    manifest.update(
        {
            "stage": "dataset_built",
            "dataset_config_version": DATASET_CONFIG_VERSION,
            "selection": config.selection,
            "dataset": {
                "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "n_matches": len(matches),
                "match_ids_sha256": match_ids_hash,
                "n_events_total": audit["n_events_total"],
                "n_shot_events": audit["n_shot_events"],
                "n_all_eligible_shots": audit["n_all_eligible"],
                "n_context_eligible_shots": audit["n_context_eligible"],
                "share_context_of_all": audit["share_context_of_all"],
                "share_with_freeze_frame": audit["share_with_freeze_frame"],
                "share_with_goalkeeper": audit["share_with_goalkeeper"],
                "share_with_statsbomb_xg": audit["share_with_statsbomb_xg"],
                "goal_rate": audit["goal_rate_all_eligible"],
                "filter_log": audit["filter_log"],
                "files": {
                    "all_eligible_shots": str(all_path.relative_to(config_root())),
                    "context_eligible_shots": str(context_path.relative_to(config_root())),
                    "all_eligible_sha256": _file_sha256(all_path),
                    "context_eligible_sha256": _file_sha256(context_path),
                },
            },
        }
    )
    DATA_MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    logger.info("Манифест обновлён: %s", DATA_MANIFEST_PATH)


def config_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
