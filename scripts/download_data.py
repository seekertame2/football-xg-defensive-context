"""Воспроизводимая избирательная загрузка StatsBomb Open Data.

Скрипт никогда не клонирует источник целиком: полный объём — около 16 ГБ.
Метаданные скачиваются целиком, это около 7 МБ.
События и файлы 360 скачиваются только для явно заданного набора матчей.
Объём известен заранее, а скачанное попадает в локальный кеш.

Примеры
-------
Метаданные и оценка объёма без загрузки событий::

    python scripts/download_data.py --config configs/data.yaml --metadata-only

Загрузка событий утверждённой выборки из configs/data.yaml::

    python scripts/download_data.py --config configs/data.yaml --selection

Загрузка событий конкретного сезона::

    python scripts/download_data.py --competition 43 --season 3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xg_context.config import (
    DATA_MANIFEST_PATH,
    DATASET_CONFIG_VERSION,
    load_data_config,
)
from xg_context.data import (
    build_downloader,
    download_all_metadata,
    download_match_files,
    estimate_download_size,
    fetch_source_inventory,
)

logger = logging.getLogger("download_data")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/data.yaml", help="путь к конфигурации данных")
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="скачать только competitions.json и файлы матчей (около 7 МБ)",
    )
    parser.add_argument(
        "--selection",
        action="store_true",
        help="скачать события утверждённой выборки из секции selection конфигурации",
    )
    parser.add_argument("--competition", type=int, default=None, help="competition_id для загрузки")
    parser.add_argument("--season", type=int, default=None, help="season_id для загрузки")
    parser.add_argument(
        "--include-three-sixty",
        action="store_true",
        help="дополнительно скачать файлы StatsBomb 360 (значительно больше по объёму)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="только оценить объём загрузки, ничего не скачивая",
    )
    parser.add_argument(
        "--max-gb",
        type=float,
        default=5.0,
        help="предохранитель: отказаться от загрузки больше указанного объёма",
    )
    parser.add_argument("--force", action="store_true", help="игнорировать локальный кеш")
    return parser.parse_args(argv)


def _resolve_selection(args: argparse.Namespace, config) -> list[tuple[int, int]]:
    """Определить, какие competition-season нужно загрузить."""
    if args.competition is not None and args.season is not None:
        return [(args.competition, args.season)]

    if args.selection:
        selection = config.selection or {}
        if not selection.get("approved", False):
            raise SystemExit(
                "Выборка в configs/data.yaml ещё не утверждена (selection.approved = false).\n"
                "Сначала прочитайте reports/data_audit.md, затем впишите утверждённые\n"
                "competition_seasons и поставьте approved: true."
            )
        pairs = [
            (int(item["competition_id"]), int(item["season_id"]))
            for item in selection.get("competition_seasons", [])
        ]
        if not pairs:
            raise SystemExit("selection.competition_seasons пуст — нечего загружать.")
        return pairs

    return []


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args(argv)
    config = load_data_config(args.config)
    downloader = build_downloader(config)

    logger.info("Источник: %s", config.source.permalink)
    logger.info("Кеш: %s", downloader.local_path("").parent)

    logger.info("Шаг 1/3: метаданные (competitions.json и файлы матчей)")
    competitions, matches_by_season = download_all_metadata(downloader)
    logger.info(
        "Метаданные готовы: %d competition-season, %d матчей",
        len(competitions),
        sum(len(v) for v in matches_by_season.values()),
    )

    logger.info("Шаг 2/3: инвентаризация источника через git-tree")
    inventory = fetch_source_inventory(downloader, force=args.force)
    logger.info(
        "В источнике: %d файлов событий, %d файлов 360",
        len(inventory.match_ids_with_events()),
        len(inventory.match_ids_with_three_sixty()),
    )

    if args.metadata_only:
        logger.info("Режим --metadata-only: события не загружаются.")
        _write_stub_manifest(config, competitions, matches_by_season, inventory, downloader)
        return 0

    pairs = _resolve_selection(args, config)
    if not pairs:
        logger.warning(
            "Не задано, что загружать. Укажите --selection, либо --competition и --season, "
            "либо используйте --metadata-only."
        )
        return 1

    match_ids: list[int] = []
    for competition_id, season_id in pairs:
        matches = matches_by_season.get((competition_id, season_id))
        if matches is None:
            raise SystemExit(f"В источнике нет competition {competition_id}, season {season_id}.")
        match_ids.extend(int(m["match_id"]) for m in matches)

    estimate = estimate_download_size(
        inventory,
        match_ids,
        include_events=True,
        include_three_sixty=args.include_three_sixty,
    )
    logger.info(
        "Шаг 3/3: к загрузке %d файлов, %d матчей, оценка объёма %.1f МБ (%.2f ГБ)",
        estimate["n_files"],
        estimate["n_matches"],
        estimate["total_mb"],
        estimate["total_gb"],
    )

    if estimate["total_gb"] > args.max_gb:
        raise SystemExit(
            f"Оценка объёма {estimate['total_gb']:.2f} ГБ превышает лимит --max-gb={args.max_gb}. "
            "Сузьте выборку или явно поднимите лимит."
        )

    if args.dry_run:
        logger.info("Режим --dry-run: загрузка не выполняется.")
        return 0

    download_match_files(
        downloader,
        match_ids,
        inventory=inventory,
        include_events=True,
        include_three_sixty=args.include_three_sixty,
    )
    logger.info(
        "Готово: скачано %d файлов (%.1f МБ), взято из кеша %d.",
        downloader.files_downloaded,
        downloader.bytes_downloaded / 1e6,
        downloader.files_from_cache,
    )
    return 0


def _write_stub_manifest(config, competitions, matches_by_season, inventory, downloader) -> None:
    """Записать манифест уровня метаданных.

    Полный манифест с числом ударов и долей защитного контекста дополняется при построении датасета.
    """
    manifest = {
        "source_url": config.source.source_url,
        "source_permalink": config.source.permalink,
        "commit_sha": config.source.commit_sha,
        "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset_config_version": DATASET_CONFIG_VERSION,
        "stage": "metadata_only",
        "n_competition_seasons": len(competitions),
        "n_matches_in_metadata": sum(len(v) for v in matches_by_season.values()),
        "n_match_event_files_in_source": len(inventory.match_ids_with_events()),
        "n_match_360_files_in_source": len(inventory.match_ids_with_three_sixty()),
        "selection": config.selection,
        "note": (
            "Числа ударов, доля защитного контекста и хеши датасета добавляются "
            "скриптом scripts/build_dataset.py."
        ),
    }
    DATA_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    logger.info("Манифест записан: %s", DATA_MANIFEST_PATH)


if __name__ == "__main__":
    raise SystemExit(main())
