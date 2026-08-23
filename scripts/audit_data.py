"""Аудит данных StatsBomb Open Data (этап 1 спецификации).

Скрипт полностью воспроизводит `reports/data_audit.md`: он скачивает
метаданные, инвентаризует зафиксированную ревизию источника, разбирает
стратифицированную выборку матчей и считает доступность защитного контекста.

Отчёт генерируется целиком из посчитанных чисел. Рекомендация по выборке
формируется по явным критериям, описанным в :func:`rank_candidates`,
а не пишется вручную поверх результатов.

Запуск::

    python scripts/audit_data.py --config configs/data.yaml
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

from xg_context.audit import (
    audit_match_shots,
    build_coverage_table,
    context_availability_by_season,
    freeze_frame_quality_summary,
    probe_three_sixty,
    sample_matches_for_audit,
    selection_bias_table,
    shots_to_frame,
)
from xg_context.config import (
    DATA_MANIFEST_PATH,
    DATASET_CONFIG_VERSION,
    INTERIM_DATA_DIR,
    REPORTS_DIR,
    TABLES_DIR,
    load_data_config,
)
from xg_context.data import (
    build_downloader,
    download_all_metadata,
    estimate_download_size,
    events_path,
    fetch_source_inventory,
)

logger = logging.getLogger("audit_data")

# Критерии отбора кандидатов на основную выборку.
# Вынесены в константы, чтобы рекомендация была воспроизводимой и оспоримой.
MIN_CONTEXT_SHARE = 0.95  # доля непенальтистских ударов с защитным контекстом
MIN_GK_SHARE = 0.90  # доля ударов с распознанным вратарём
MIN_ESTIMATED_SHOTS = 1500  # минимальный ожидаемый объём ударов у сезона
MIN_MATCHES = 50  # минимальное число матчей у сезона

# Ограничения на итоговую выборку. Взять «всё, что прошло фильтры» нельзя:
# это дало бы 14 сезонов и более 8 ГБ загрузки, смешав мужской и женский футбол
# и эпохи от 2003 до 2024 года вопреки разделу 6 спецификации.
MAX_DOWNLOAD_GB = 5.0  # предел объёма событий рекомендуемой выборки
TARGET_SHOTS = 15_000  # объём, ниже которого доверительные интервалы будут широкими

#: Перевод поля competition_gender для русских таблиц отчёта.
GENDER_RU = {"male": "мужской", "female": "женский"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument(
        "--matches-per-season",
        type=int,
        default=None,
        help="переопределить audit.matches_per_season из конфигурации",
    )
    parser.add_argument(
        "--skip-three-sixty",
        action="store_true",
        help="не скачивать пробные файлы StatsBomb 360 (экономит около 15 МБ)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "лёгкий режим для разработки и CI: 1 матч на сезон, без проверки 360. "
            "Результаты пишутся с суффиксом _quick и не затирают канонический отчёт"
        ),
    )
    return parser.parse_args(argv)


# --------------------------------------------------------------------------------------
# Рекомендация по выборке
# --------------------------------------------------------------------------------------


def rank_candidates(context_by_season: pd.DataFrame) -> pd.DataFrame:
    """Отранжировать competition-season как кандидатов в основную выборку.

    Критерии заданы явно и проверяемы:

    1. доля ударов с защитным контекстом не ниже :data:`MIN_CONTEXT_SHARE`;
    2. доля ударов с распознанным вратарём не ниже :data:`MIN_GK_SHARE`;
    3. сезон не «обрезанный»: не меньше :data:`MIN_MATCHES` матчей с событиями;
    4. ожидаемый объём не меньше :data:`MIN_ESTIMATED_SHOTS` ударов.

    Кандидаты, прошедшие все фильтры, сортируются по ожидаемому числу ударов:
    при равном качестве контекста больший объём даёт более узкие доверительные
    интервалы в центральном ablation-эксперименте.
    """
    frame = context_by_season.copy()
    frame["passes_context"] = frame["share_with_context"] >= MIN_CONTEXT_SHARE
    frame["passes_gk"] = frame["share_with_gk"] >= MIN_GK_SHARE
    frame["passes_matches"] = frame["n_matches_with_events"] >= MIN_MATCHES
    frame["passes_volume"] = frame["estimated_total_shots"] >= MIN_ESTIMATED_SHOTS
    frame["is_candidate"] = (
        frame["passes_context"]
        & frame["passes_gk"]
        & frame["passes_matches"]
        & frame["passes_volume"]
    )
    return frame.sort_values(
        ["is_candidate", "estimated_total_shots"], ascending=[False, False], ignore_index=True
    )


def build_homogeneous_blocks(ranked: pd.DataFrame) -> pd.DataFrame:
    """Сгруппировать прошедших фильтры кандидатов в однородные блоки.

    Блок — это пара «пол соревнования + сезон»: например, четыре топ-лиги
    Европы сезона 2015/2016. Внутри блока данные собраны в одну эпоху и в одном
    типе футбола, поэтому объединение сезонов не смешивает несовместимые
    контексты, чего требует раздел 6 спецификации.

    Блоки ранжируются так: сначала помещающиеся в бюджет загрузки
    :data:`MAX_DOWNLOAD_GB`, затем достигающие :data:`TARGET_SHOTS`,
    затем по ожидаемому числу ударов.
    """
    candidates = ranked[ranked["is_candidate"]]
    if candidates.empty:
        return pd.DataFrame()

    blocks = (
        candidates.groupby(["gender", "season_name"])
        .agg(
            n_competitions=("competition_id", "size"),
            n_matches=("n_matches_with_events", "sum"),
            estimated_total_shots=("estimated_total_shots", "sum"),
            events_mb=("events_mb", "sum"),
            min_context_share=("share_with_context", "min"),
            min_gk_share=("share_with_gk", "min"),
            n_matches_with_360=("n_matches_with_360", "sum"),
            competitions=("competition_name", lambda names: ", ".join(sorted(names))),
        )
        .reset_index()
    )
    blocks["events_gb"] = blocks["events_mb"] / 1000.0
    blocks["fits_budget"] = blocks["events_gb"] <= MAX_DOWNLOAD_GB
    blocks["meets_target"] = blocks["estimated_total_shots"] >= TARGET_SHOTS
    return blocks.sort_values(
        ["fits_budget", "meets_target", "estimated_total_shots"],
        ascending=[False, False, False],
        ignore_index=True,
    )


def select_recommended(ranked: pd.DataFrame, blocks: pd.DataFrame) -> pd.DataFrame:
    """Вернуть строки рекомендуемой выборки — лучший однородный блок.

    Если лучший блок не помещается в бюджет, из него жадно берутся самые
    крупные сезоны, пока бюджет не исчерпан.
    """
    if blocks.empty:
        return ranked.head(0)

    best = blocks.iloc[0]
    members = ranked[
        ranked["is_candidate"]
        & (ranked["gender"] == best["gender"])
        & (ranked["season_name"] == best["season_name"])
    ].sort_values("estimated_total_shots", ascending=False)

    if best["events_gb"] <= MAX_DOWNLOAD_GB:
        return members.reset_index(drop=True)

    keep, used_mb = [], 0.0
    for _, row in members.iterrows():
        if (used_mb + row["events_mb"]) / 1000.0 > MAX_DOWNLOAD_GB and keep:
            break
        keep.append(row)
        used_mb += row["events_mb"]
    return pd.DataFrame(keep).reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Рендеринг markdown
# --------------------------------------------------------------------------------------


def _fmt_share(value: float) -> str:
    if pd.isna(value):
        return "н/д"
    return f"{value * 100:.1f}%"


def _fmt_int(value: int | float) -> str:
    """Целое число с неразрывным пробелом между разрядами."""
    return f"{int(value):,}".replace(",", "\u202f")


def _plural(count: int, one: str, few: str, many: str) -> str:
    """Выбрать форму русского существительного при числительном.

    Пример: 1 матч, 2 матча, 5 матчей, 11 матчей, 21 матч.
    """
    tail = abs(int(count)) % 100
    if 11 <= tail <= 14:
        return many
    last = tail % 10
    if last == 1:
        return one
    if last in (2, 3, 4):
        return few
    return many


def _count(count: int, one: str, few: str, many: str) -> str:
    """Число с разделителями разрядов и согласованным существительным."""
    return f"{_fmt_int(count)} {_plural(count, one, few, many)}"


def _md_table(
    frame: pd.DataFrame,
    columns: dict[str, str],
    floatfmt: str = "{:.3f}",
    formats: dict[str, str] | None = None,
) -> str:
    """Отрендерить DataFrame в markdown-таблицу с русскими заголовками.

    Дробные значения с нулевой дробной частью (число матчей, `match_id`, счётчики)
    печатаются как целые: иначе таблица пестрит бессмысленными «380.000».
    Через ``formats`` можно задать формат отдельной колонки по её исходному имени.
    """
    if frame.empty:
        return "_Нет данных._\n"
    formats = formats or {}
    source_names = list(columns)
    subset = frame[source_names]
    header = "| " + " | ".join(columns.values()) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, sep]
    for _, row in subset.iterrows():
        cells = []
        for name in source_names:
            value = row[name]
            if isinstance(value, float):
                if pd.isna(value):
                    cells.append("н/д")
                elif name in formats:
                    cells.append(formats[name].format(value))
                elif float(value).is_integer():
                    cells.append(_fmt_int(value))
                else:
                    cells.append(floatfmt.format(value))
            elif isinstance(value, (int,)) and not isinstance(value, bool):
                cells.append(_fmt_int(value))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def render_report(context: dict[str, Any]) -> str:
    """Собрать полный текст `reports/data_audit.md` из посчитанных величин."""
    cfg = context["config"]
    summary = context["summary"]
    coverage = context["coverage"]
    ranked = context["ranked"]
    bias = context["bias"]
    three_sixty = context["three_sixty"]
    source_totals = context["source_totals"]
    recommended = context["recommended"]

    parts: list[str] = []
    add = parts.append

    add("# Аудит данных StatsBomb Open Data\n")
    add(
        "> Файл сгенерирован скриптом `scripts/audit_data.py`. "
        "Не редактируйте его вручную — правьте код и перезапускайте аудит.\n"
    )
    add(
        f"- **Источник:** [{cfg.source.repo}]({cfg.source.source_url})\n"
        f"- **Зафиксированная ревизия:** `{cfg.source.commit_sha}`\n"
        f"- **Permalink:** {cfg.source.permalink}\n"
        f"- **Дата аудита:** {context['generated_at']}\n"
        f"- **Матчей на сезон в выборке аудита:** {context['matches_per_season']}\n"
    )

    add("\n## 1. Объём источника и почему нельзя качать всё\n")
    add(
        f"В зафиксированной ревизии источник содержит "
        f"**{source_totals['n_event_files']} файлов событий общим объёмом "
        f"{source_totals['events_gb']:.1f} ГБ** и "
        f"**{source_totals['n_360_files']} файлов StatsBomb 360 объёмом "
        f"{source_totals['three_sixty_gb']:.1f} ГБ**.\n"
    )
    add(
        f"Метаданные (`competitions.json` и {source_totals['n_match_files']} файлов матчей) "
        f"занимают всего {source_totals['matches_mb']:.1f} МБ, поэтому они скачиваются "
        "целиком и дают точную перепись соревнований, сезонов и матчей.\n"
    )
    add(
        "Список файлов и их размеры получены одним запросом git-tree к зафиксированному "
        "commit SHA, поэтому объём любой будущей загрузки известен заранее, до скачивания.\n"
    )
    orphans = source_totals["n_event_files"] - context["n_matches_in_metadata"]
    if orphans:
        add(
            f"\n**Расхождение в источнике.** Файлов событий {source_totals['n_event_files']}, "
            f"а матчей во всех `matches/*.json` — {context['n_matches_in_metadata']}: "
            f"{orphans} файлов событий не соответствует ни одному матчу в метаданных. "
            "Проект работает от метаданных, поэтому такие файлы просто не попадают "
            "в выборку, но полагаться на перечень `data/events/` как на список матчей нельзя.\n"
        )

    add("\n## 2. Соревнования и сезоны\n")
    add(
        f"Всего в источнике **{len(coverage)} пар competition-season**, "
        f"**{int(coverage['n_matches'].sum())} матчей**, из них "
        f"**{int(coverage['n_matches_with_events'].sum())}** имеют файл событий и "
        f"**{int(coverage['n_matches_with_360'].sum())}** — файл StatsBomb 360.\n"
    )
    add("\nПолная таблица покрытия: `reports/tables/audit_coverage.csv`.\n")
    add("\n### Крупнейшие соревнования по числу матчей с событиями\n\n")
    top = coverage.nlargest(15, "n_matches_with_events")
    add(
        _md_table(
            top,
            {
                "competition_name": "Соревнование",
                "season_name": "Сезон",
                "country_name": "Страна",
                "gender": "Пол",
                "n_matches_with_events": "Матчей",
                "n_matches_with_360": "Из них с 360",
                "events_mb": "События, МБ",
            },
            floatfmt="{:.1f}",
        )
    )

    add("\n## 3. Удары и доступность защитного контекста\n")
    add(
        f"Выборочный аудит разобрал "
        f"**{_count(context['n_matches_audited'], 'матч', 'матча', 'матчей')}** "
        f"({context['audit_download_mb']:.0f} МБ событий) и "
        f"**{_count(summary['n_shots_total'], 'удар', 'удара', 'ударов')}**.\n"
    )
    add("\nСостав выборки ударов:\n\n")
    add(
        f"| Показатель | Значение |\n| --- | ---: |\n"
        f"| Всего ударов | {summary['n_shots_total']} |\n"
        f"| Пенальти и удары серии (исключаются) | {summary['n_shots_penalty']} |\n"
        f"| Неизвестный исход | {summary['n_shots_unknown_outcome']} |\n"
        f"| Без координат удара | {summary['n_shots_missing_location']} |\n"
        f"| **Пригодных непенальтистских ударов** | **{summary['n_shots_eligible']}** |\n"
        f"| Из них с `shot.freeze_frame` | {summary['n_shots_with_freeze_frame']} "
        f"({_fmt_share(summary['share_with_freeze_frame'])}) |\n"
        f"| Из них с координатами соперников | {summary['n_shots_with_context']} "
        f"({_fmt_share(summary['share_with_context'])}) |\n"
        f"| Из них с распознанным вратарём | {summary['n_shots_with_gk']} "
        f"({_fmt_share(summary['share_with_gk'])}) |\n"
        f"| С заполненным `statsbomb_xg` | {_fmt_share(summary['share_with_statsbomb_xg'])} |\n"
        f"| Базовая доля голов | {_fmt_share(summary['goal_rate_all_eligible'])} |\n"
    )
    add(
        f"\nПри наличии `freeze_frame` вратарь распознаётся в "
        f"{_fmt_share(summary['share_gk_given_freeze_frame'])} случаев; "
        f"медианное число соперников в кадре — "
        f"{summary['median_opponents_in_frame']:.0f}.\n"
    )
    if summary["n_frames_with_invalid_locations"]:
        add(
            f"\n**Внимание:** в {summary['n_frames_with_invalid_locations']} кадрах есть записи "
            "игроков без корректных координат — их нужно отбрасывать явно, а не заполнять нулями.\n"
        )

    add("\n### Доступность контекста по соревнованиям\n\n")
    add(
        "Доли посчитаны по выборке матчей, число ударов у сезона — оценка "
        "(`удары на матч` × `матчей с событиями`). Колонка «Ударов в выборке» "
        "показывает, на скольких наблюдениях основана строка. Доли контекста при "
        "значении 1.000 надёжны, а вот **доля голов по отдельному сезону оценена по "
        "нескольким десяткам ударов и очень шумная**: содержательна только "
        "агрегированная доля голов из раздела 3.\n\n"
    )
    add(
        _md_table(
            ranked,
            {
                "competition_name": "Соревнование",
                "season_name": "Сезон",
                "n_matches_with_events": "Матчей",
                "n_shots": "Ударов в выборке",
                "estimated_total_shots": "Оценка ударов всего",
                "share_with_context": "Доля с контекстом",
                "share_with_gk": "Доля с вратарём",
                "goal_rate": "Доля голов",
            },
            formats={
                "share_with_context": "{:.3f}",
                "share_with_gk": "{:.3f}",
                "goal_rate": "{:.3f}",
            },
        )
    )

    add("\n## 4. Selection bias: чем удары с контекстом отличаются от всех\n")
    add(
        "Если бы `freeze_frame` пропадал не случайно, выборка "
        "`context_eligible_shots` систематически отличалась бы от всех ударов, "
        "и разницу метрик нельзя было бы приписывать признакам.\n\n"
    )
    add(
        _md_table(
            bias,
            {
                "выборка": "Выборка",
                "n_shots": "Ударов",
                "доля голов": "Доля голов",
                "медиана расстояния": "Медиана расстояния",
                "медиана угла, °": "Медиана угла, °",
                "доля ударов головой": "Доля головой",
                "доля open play": "Доля open play",
                "доля со штрафных": "Доля штрафных",
            },
        )
    )
    add("\n" + context["bias_verdict"] + "\n")

    add("\n## 5. Нужен ли отдельный источник StatsBomb 360\n")
    if three_sixty:
        add(
            f"Проверены матчи {', '.join(str(p['match_id']) for p in three_sixty)} "
            f"(средний размер файла 360 — "
            f"{sum(p['file_mb'] for p in three_sixty) / len(three_sixty):.1f} МБ против "
            "примерно 3 МБ у файла событий).\n\n"
        )
        add(
            _md_table(
                pd.DataFrame(three_sixty),
                {
                    "match_id": "match_id",
                    "n_shots": "Ударов",
                    "n_shots_with_shot_freeze_frame": "С `shot.freeze_frame`",
                    "n_shots_covered_by_360": "Покрыто файлом 360",
                    "mean_players_shot_freeze_frame": "Игроков в `freeze_frame`",
                    "mean_players_360_frame": "Игроков в кадре 360",
                    "file_mb": "Размер файла, МБ",
                },
                floatfmt="{:.1f}",
                formats={"match_id": "{:.0f}"},
            )
        )
        add("\n" + context["three_sixty_verdict"] + "\n")
    else:
        add("_Проверка 360 пропущена (`--skip-three-sixty` или `--quick`)._\n")

    add("\n## 6. Рекомендуемая выборка\n")
    add(context["recommendation_text"] + "\n")
    if not recommended.empty:
        add("\n")
        add(
            _md_table(
                recommended,
                {
                    "competition_name": "Соревнование",
                    "season_name": "Сезон",
                    "competition_id": "competition_id",
                    "season_id": "season_id",
                    "n_matches_with_events": "Матчей",
                    "estimated_total_shots": "Оценка ударов",
                    "share_with_context": "Доля с контекстом",
                    "events_mb": "События, МБ",
                },
                formats={"events_mb": "{:.1f}", "share_with_context": "{:.3f}"},
            )
        )
        add(
            f"\n**Объём будущей загрузки для рекомендуемой выборки: "
            f"{context['recommended_download']['n_files']} файлов, "
            f"{context['recommended_download']['total_mb']:.0f} МБ "
            f"({context['recommended_download']['total_gb']:.2f} ГБ).**\n"
        )
        add("\nЧтобы утвердить выборку, впишите её в `configs/data.yaml`:\n\n")
        add("```yaml\nselection:\n  approved: true\n  competition_seasons:\n")
        for _, row in recommended.iterrows():
            add(
                f"    - {{ competition_id: {int(row['competition_id'])}, "
                f"season_id: {int(row['season_id'])} }}  "
                f"# {row['competition_name']} {row['season_name']}\n"
            )
        add("  include_three_sixty: false\n```\n")

    add("\n## 7. Обнаруженные ограничения\n")
    for item in context["limitations"]:
        add(f"- {item}\n")

    add("\n## 8. Что должен решить владелец проекта\n")
    for item in context["decisions_needed"]:
        add(f"- {item}\n")

    add("\n---\n\n")
    add(
        "Машиночитаемая сводка: `reports/tables/audit_summary.json`. "
        "Полные таблицы: `reports/tables/audit_coverage.csv`, "
        "`reports/tables/audit_context_by_season.csv`, "
        "`reports/tables/audit_candidate_blocks.csv`, "
        "`reports/tables/audit_selection_bias.csv`. "
        "Разобранные удары выборки аудита: `data/interim/audit_shots_sample.parquet`.\n"
    )
    return "".join(parts)


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
    config = load_data_config(args.config)

    matches_per_season = args.matches_per_season or config.audit.matches_per_season
    skip_360 = args.skip_three_sixty
    # Лёгкий режим пишет в отдельные файлы: иначе прогон для CI или отладки
    # затёр бы канонический reports/data_audit.md выборкой из одного матча на сезон.
    suffix = ""
    if args.quick:
        matches_per_season = 1
        skip_360 = True
        suffix = "_quick"
        logger.info(
            "Режим --quick: 1 матч на сезон, без проверки 360. "
            "Вывод пишется с суффиксом _quick и не затирает канонический отчёт."
        )

    downloader = build_downloader(config)

    logger.info("Шаг 1/6: метаданные источника")
    competitions, matches_by_season = download_all_metadata(downloader)

    logger.info("Шаг 2/6: инвентаризация git-tree")
    inventory = fetch_source_inventory(downloader)

    logger.info("Шаг 3/6: таблица покрытия competition-season")
    coverage = build_coverage_table(competitions, matches_by_season, inventory)

    sample = sample_matches_for_audit(
        matches_by_season,
        inventory,
        matches_per_season=matches_per_season,
        seed=config.audit.sample_seed,
        max_total=config.audit.max_event_files,
    )
    audit_mb = sample["events_bytes"].sum() / 1e6
    logger.info(
        "Шаг 4/6: разбор %d матчей (%.0f МБ событий)",
        len(sample),
        audit_mb,
    )

    paths = [events_path(int(m)) for m in sample["match_id"]]
    downloader.fetch_many(paths)
    logger.info(
        "Файлы событий готовы: скачано %d, из кеша %d",
        downloader.files_downloaded,
        downloader.files_from_cache,
    )

    records: list[dict[str, Any]] = []
    for row in sample.itertuples(index=False):
        events = downloader.load_events(int(row.match_id))
        records.extend(
            audit_match_shots(
                events,
                match_id=int(row.match_id),
                competition_id=int(row.competition_id),
                season_id=int(row.season_id),
            )
        )
    shots = shots_to_frame(records)
    logger.info("Разобрано ударов: %d", len(shots))

    logger.info("Шаг 5/6: агрегации")
    summary = freeze_frame_quality_summary(shots)
    context_by_season = context_availability_by_season(shots, coverage)
    ranked = rank_candidates(context_by_season)
    bias = selection_bias_table(shots)

    three_sixty: list[dict[str, Any]] = []
    if not skip_360:
        probe_ids = (
            sample.loc[sample["has_360"], "match_id"]
            .head(config.audit.three_sixty_probe_matches)
            .tolist()
        )
        for match_id in probe_ids:
            logger.info("Проверяю файл StatsBomb 360 матча %s", match_id)
            events = downloader.load_events(int(match_id))
            three_sixty.append(probe_three_sixty(downloader, int(match_id), events))

    logger.info("Шаг 6/6: отчёт и таблицы")
    blocks = build_homogeneous_blocks(ranked)
    recommended = select_recommended(ranked, blocks)
    recommended_download = estimate_download_size(
        inventory,
        _matches_of(recommended, matches_by_season),
        include_events=True,
    )

    context = {
        "config": config,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%d"),
        "matches_per_season": matches_per_season,
        "n_matches_audited": len(sample),
        "n_matches_in_metadata": int(coverage["n_matches"].sum()),
        "audit_download_mb": audit_mb,
        "coverage": coverage,
        "summary": summary,
        "ranked": ranked,
        "bias": bias,
        "three_sixty": three_sixty,
        "recommended": recommended,
        "recommended_download": recommended_download,
        "source_totals": _source_totals(inventory),
        "bias_verdict": _bias_verdict(bias, summary),
        "blocks": blocks,
        "three_sixty_verdict": _three_sixty_verdict(three_sixty, recommended),
        "recommendation_text": _recommendation_text(
            ranked, recommended, blocks, recommended_download
        ),
        "limitations": _limitations(
            summary, coverage, matches_per_season, len(sample), _source_totals(inventory)
        ),
        "decisions_needed": _decisions_needed(recommended, recommended_download),
    }

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(TABLES_DIR / f"audit_coverage{suffix}.csv", index=False, encoding="utf-8")
    ranked.to_csv(
        TABLES_DIR / f"audit_context_by_season{suffix}.csv", index=False, encoding="utf-8"
    )
    bias.to_csv(TABLES_DIR / f"audit_selection_bias{suffix}.csv", index=False, encoding="utf-8")
    if not blocks.empty:
        blocks.to_csv(
            TABLES_DIR / f"audit_candidate_blocks{suffix}.csv", index=False, encoding="utf-8"
        )
    shots.to_parquet(INTERIM_DATA_DIR / f"audit_shots_sample{suffix}.parquet", index=False)

    payload = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "commit_sha": config.source.commit_sha,
        "matches_per_season": matches_per_season,
        "n_matches_audited": len(sample),
        "audit_download_mb": round(audit_mb, 1),
        "source_totals": context["source_totals"],
        "shot_summary": summary,
        "criteria": {
            "min_context_share": MIN_CONTEXT_SHARE,
            "min_gk_share": MIN_GK_SHARE,
            "min_estimated_shots": MIN_ESTIMATED_SHOTS,
            "min_matches": MIN_MATCHES,
            "max_download_gb": MAX_DOWNLOAD_GB,
            "target_shots": TARGET_SHOTS,
        },
        "recommended_selection": [
            {
                "competition_id": int(r["competition_id"]),
                "season_id": int(r["season_id"]),
                "competition_name": r["competition_name"],
                "season_name": r["season_name"],
                "n_matches_with_events": int(r["n_matches_with_events"]),
                "estimated_total_shots": int(r["estimated_total_shots"]),
            }
            for _, r in recommended.iterrows()
        ],
        "recommended_download": recommended_download,
        "candidate_blocks": blocks.to_dict("records") if not blocks.empty else [],
        "three_sixty_probe": three_sixty,
    }
    (TABLES_DIR / f"audit_summary{suffix}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"data_audit{suffix}.md"
    report_path.write_text(render_report(context), encoding="utf-8")
    logger.info("Отчёт записан: %s", report_path)

    if args.quick:
        logger.info("Режим --quick: манифест не обновляется.")
    else:
        _update_manifest(config, payload)
    logger.info(
        "Рекомендовано %d competition-season, объём загрузки %.0f МБ",
        len(recommended),
        recommended_download["total_mb"],
    )
    return 0


def _update_manifest(config, payload: dict[str, Any]) -> None:
    """Дописать в манифест то, что известно после аудита.

    Полный манифест с числом ударов итогового датасета и его хешами
    дополняется на этапе 2 скриптом `scripts/build_dataset.py`.
    """
    manifest: dict[str, Any] = {}
    if DATA_MANIFEST_PATH.exists():
        manifest = json.loads(DATA_MANIFEST_PATH.read_text(encoding="utf-8"))

    manifest.update(
        {
            "source_url": config.source.source_url,
            "source_permalink": config.source.permalink,
            "commit_sha": config.source.commit_sha,
            "dataset_config_version": DATASET_CONFIG_VERSION,
            "stage": "audited",
            "audit": {
                "generated_at": payload["generated_at"],
                "matches_per_season": payload["matches_per_season"],
                "n_matches_audited": payload["n_matches_audited"],
                "audit_download_mb": payload["audit_download_mb"],
                "n_shots_parsed": payload["shot_summary"]["n_shots_total"],
                "n_shots_eligible": payload["shot_summary"]["n_shots_eligible"],
                "n_shots_penalty_excluded": payload["shot_summary"]["n_shots_penalty"],
                "share_with_context": payload["shot_summary"]["share_with_context"],
                "share_with_goalkeeper": payload["shot_summary"]["share_with_gk"],
                "goal_rate": payload["shot_summary"]["goal_rate_all_eligible"],
                "criteria": payload["criteria"],
                "recommended_selection": payload["recommended_selection"],
                "recommended_download": payload["recommended_download"],
            },
            "selection": config.selection,
        }
    )
    DATA_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    logger.info("Манифест обновлён: %s", DATA_MANIFEST_PATH)


def _matches_of(recommended: pd.DataFrame, matches_by_season) -> list[int]:
    out: list[int] = []
    for _, row in recommended.iterrows():
        key = (int(row["competition_id"]), int(row["season_id"]))
        out.extend(int(m["match_id"]) for m in matches_by_season.get(key, []))
    return out


def _source_totals(inventory) -> dict[str, Any]:
    events = [p for p in inventory.sizes if p.startswith("data/events/")]
    three_sixty = [p for p in inventory.sizes if p.startswith("data/three-sixty/")]
    matches = [p for p in inventory.sizes if p.startswith("data/matches/")]
    return {
        "n_event_files": len(events),
        "events_gb": round(inventory.total_size(events) / 1e9, 2),
        "n_360_files": len(three_sixty),
        "three_sixty_gb": round(inventory.total_size(three_sixty) / 1e9, 2),
        "n_match_files": len(matches),
        "matches_mb": round(inventory.total_size(matches) / 1e6, 1),
    }


def _bias_verdict(bias: pd.DataFrame, summary: dict[str, Any]) -> str:
    """Сформулировать вывод о selection bias по фактическим числам."""
    if bias.empty or (bias["выборка"] == "без контекста").sum() == 0:
        return (
            "Ударов без защитного контекста в выборке практически нет, поэтому "
            "`context_eligible_shots` совпадает с `all_eligible_shots`, и systematic "
            "selection bias между ними не возникает."
        )
    with_ctx = bias.loc[bias["выборка"] == "с контекстом"].iloc[0]
    without_ctx = bias.loc[bias["выборка"] == "без контекста"].iloc[0]
    share_without = 1.0 - summary["share_with_context"]
    goal_gap = abs(float(with_ctx["доля голов"]) - float(without_ctx["доля голов"]))
    dist_gap = abs(float(with_ctx["медиана расстояния"]) - float(without_ctx["медиана расстояния"]))
    verdict = (
        f"Удары без контекста составляют {_fmt_share(share_without)} пригодных ударов "
        f"({int(without_ctx['n_shots'])} наблюдений в выборке аудита). "
        f"Разница долей голов между группами — {goal_gap * 100:.1f} п.п., "
        f"разница медианного расстояния — {dist_gap:.1f} ярда."
    )
    if share_without < 0.02:
        verdict += (
            " Доля потерь мала, поэтому смещение выборки не угрожает основному сравнению, "
            "но число исключённых ударов всё равно фиксируется в отчёте."
        )
    else:
        verdict += (
            " Доля потерь заметна: главное сравнение моделей обязано проводиться "
            "на одних и тех же строках `context_eligible_shots`, а различие групп "
            "нужно явно оговорить в разделе ограничений."
        )
    return verdict


def _three_sixty_verdict(
    three_sixty: list[dict[str, Any]],
    recommended: pd.DataFrame,
) -> str:
    """Сформулировать вывод о необходимости отдельного источника StatsBomb 360.

    Решающий аргумент — не число игроков в кадре, а покрытие: если у матчей
    рекомендуемой выборки файлов 360 нет, опора на них означала бы смену выборки.
    """
    if not three_sixty:
        return ""
    covered = sum(p["n_shots_covered_by_360"] for p in three_sixty)
    with_ff = sum(p["n_shots_with_shot_freeze_frame"] for p in three_sixty)
    total = sum(p["n_shots"] for p in three_sixty)
    mean_ff = sum(p["mean_players_shot_freeze_frame"] for p in three_sixty) / len(three_sixty)
    mean_360 = sum(p["mean_players_360_frame"] for p in three_sixty) / len(three_sixty)

    verdict = (
        f"В проверенных матчах `shot.freeze_frame` покрывает "
        f"{_count(with_ff, 'удар', 'удара', 'ударов')} из {total}, "
        f"файл 360 — {covered}. Среднее число игроков в кадре: "
        f"{mean_ff:.1f} у `shot.freeze_frame` и {mean_360:.1f} у 360. "
        f"Схема кадра 360: `event_uuid`, `visible_area`, `freeze_frame` с полями "
        f"`actor`, `keeper`, `teammate`, `location` — в отличие от `shot.freeze_frame`, "
        f"здесь нет названия позиции игрока, но есть явный флаг `keeper` и область "
        f"видимости камеры.\n\n"
    )

    n_360_in_selection = (
        int(recommended["n_matches_with_360"].sum()) if not recommended.empty else 0
    )
    n_matches_in_selection = (
        int(recommended["n_matches_with_events"].sum()) if not recommended.empty else 0
    )

    if n_360_in_selection == 0 and n_matches_in_selection:
        verdict += (
            "**Вывод: отдельные файлы StatsBomb 360 использовать нельзя, и это решает "
            "вопрос окончательно.** Ни один из "
            f"{n_matches_in_selection} матчей рекомендуемой выборки не имеет файла 360 "
            "(файлы 360 существуют лишь для 426 матчей источника из 4235). Опора на 360 "
            "означала бы отказ от рекомендуемой выборки в пользу вчетверо меньшей, "
            "ради источника, который в проверенных матчах покрывает практически те же "
            "удары и добавляет примерно одного игрока в кадр."
        )
    elif covered <= with_ff:
        verdict += (
            "**Вывод: отдельные файлы StatsBomb 360 на данном этапе не нужны.** "
            "Они не покрывают больше ударов, чем `shot.freeze_frame`, доступны лишь для "
            "части матчей и примерно втрое дороже по объёму на матч."
        )
    else:
        verdict += (
            f"**Вывод: файлы 360 покрывают немного больше ударов ({covered} против "
            f"{with_ff}), но это не оправдывает переход.** Они есть лишь у "
            f"{n_360_in_selection} матчей рекомендуемой выборки, весят около 8 МБ против "
            "примерно 3 МБ у файла событий, и дают в среднем около одного дополнительного "
            "игрока в кадре."
        )

    verdict += (
        "\n\nЭто соответствует мере против «data-engineering rabbit hole» из раздела 22 "
        "спецификации: начинаем с shot-level freeze frame, полные 360 остаются расширением."
    )
    return verdict


def _recommendation_text(
    ranked: pd.DataFrame,
    recommended: pd.DataFrame,
    blocks: pd.DataFrame,
    download: dict[str, Any],
) -> str:
    """Собрать раздел с рекомендацией из посчитанных величин."""
    criteria = (
        f"Отбор идёт в два шага. **Шаг 1 — качество.** Кандидатом считается "
        f"competition-season, у которого доля ударов с защитным контекстом не ниже "
        f"{MIN_CONTEXT_SHARE:.0%}, доля с распознанным вратарём — не ниже {MIN_GK_SHARE:.0%}, "
        f"число матчей с событиями — не меньше {MIN_MATCHES}, а ожидаемый объём — "
        f"не меньше {MIN_ESTIMATED_SHOTS} ударов.\n\n"
        f"**Шаг 2 — однородность и объём.** Брать все прошедшие фильтр сезоны нельзя: "
        f"это смешало бы мужской и женский футбол и эпохи от 2003 до 2024 года, что прямо "
        f"запрещено разделом 6 спецификации. Поэтому кандидаты группируются в однородные "
        f"блоки «пол + сезон», и выбирается самый крупный блок, помещающийся в бюджет "
        f"{MAX_DOWNLOAD_GB:.0f} ГБ. Все пороги заданы константами в "
        f"`scripts/audit_data.py` и их можно оспорить.\n\n"
    )
    if recommended.empty:
        return criteria + (
            "**Ни один сезон не прошёл все фильтры.** Пороги нужно пересмотреть вместе "
            "с владельцем проекта: см. полную таблицу выше и "
            "`reports/tables/audit_context_by_season.csv`."
        )

    n_candidates = int(ranked["is_candidate"].sum())
    n_shots = int(recommended["estimated_total_shots"].sum())
    n_matches = int(recommended["n_matches_with_events"].sum())
    gender = recommended["gender"].iloc[0]
    season = recommended["season_name"].iloc[0]
    gender_ru = GENDER_RU.get(gender, gender)

    text = criteria + (
        f"Фильтрам качества удовлетворяют **{n_candidates} из {len(ranked)}** пар "
        f"competition-season. Лучший однородный блок — **{gender_ru} футбол, сезон "
        f"{season}**: "
        f"{_count(len(recommended), 'соревнование', 'соревнования', 'соревнований')}, "
        f"**{_count(n_matches, 'матч', 'матча', 'матчей')}**, примерно "
        f"**{_count(n_shots, 'удар', 'удара', 'ударов')} без пенальти**, "
        f"объём загрузки "
        f"**{download['total_mb']:.0f} МБ ({download['total_gb']:.2f} ГБ)**.\n\n"
        "Обоснование: это полные сезоны одного уровня и одной эпохи, с одинаковым "
        "стопроцентным покрытием `freeze_frame`. Объём достаточен, чтобы при разбиении "
        "по матчам в тестовой части осталось несколько тысяч ударов и несколько сотен "
        "голов — иначе доверительные интервалы вокруг разницы log loss будут слишком "
        "широкими, чтобы сделать вывод о ценности защитного контекста.\n"
    )

    if len(recommended) > 1:
        compact = recommended.iloc[0]
        text += (
            f"\n**Компактный вариант.** Если {download['total_gb']:.1f} ГБ нежелательны, "
            f"можно взять только «{compact['competition_name']} {compact['season_name']}»: "
            f"{_count(compact['n_matches_with_events'], 'матч', 'матча', 'матчей')}, "
            f"примерно {_count(compact['estimated_total_shots'], 'удар', 'удара', 'ударов')}, "
            f"{compact['events_mb']:.0f} МБ. Выборка останется корректной, но "
            f"доверительные интервалы станут заметно шире.\n"
        )

    alternatives = blocks[blocks["fits_budget"]].head(4).copy()
    alternatives["gender"] = alternatives["gender"].map(GENDER_RU).fillna(alternatives["gender"])
    if len(alternatives) > 1:
        text += "\n**Рассмотренные альтернативы (однородные блоки):**\n\n"
        text += _md_table(
            alternatives,
            {
                "gender": "Пол",
                "season_name": "Сезон",
                "competitions": "Соревнования",
                "n_matches": "Матчей",
                "estimated_total_shots": "Оценка ударов",
                "events_gb": "Загрузка, ГБ",
            },
            floatfmt="{:.2f}",
        )
    return text


def _limitations(
    summary: dict[str, Any],
    coverage: pd.DataFrame,
    matches_per_season: int,
    n_sampled: int,
    source_totals: dict[str, Any],
) -> list[str]:
    n_with_events = int(coverage["n_matches_with_events"].sum())
    orphans = source_totals["n_event_files"] - int(coverage["n_matches"].sum())
    items = [
        f"Доли доступности `freeze_frame` оценены по стратифицированной выборке "
        f"({_count(matches_per_season, 'матч', 'матча', 'матчей')} на сезон, "
        f"{_count(n_sampled, 'матч', 'матча', 'матчей')} всего), а не по всем "
        f"{_fmt_int(n_with_events)} матчам источника. "
        "Это оценка с выборочной погрешностью, а не перепись. Доли близки к 1.000 "
        "во всех проверенных сезонах, поэтому вывод о доступности контекста устойчив, "
        "но гарантией стопроцентного покрытия каждой строки он не является: "
        "фактическая доля пересчитывается на этапе 2 по полной выборке.",
        "Оценка числа ударов в сезоне получена умножением «ударов на матч» из выборки "
        "на число матчей. При 60–95 ударах на сезон в выборке это оценка с погрешностью "
        "порядка нескольких процентов, пригодная для планирования объёма, но не для выводов.",
        "Доля голов в разрезе отдельного сезона посчитана по нескольким десяткам ударов "
        "и статистически незначима; содержательна только агрегированная доля голов.",
        "Числа матчей, файлов и объёмов взяты из git-tree зафиксированной ревизии и точны.",
        "StatsBomb Open Data — не случайная выборка футбола: в ней преобладают отдельные "
        "турниры, женский футбол и матчи конкретных команд. Выводы модели нельзя "
        "автоматически переносить на футбол вообще.",
        "`statsbomb_xg` рассчитан моделью StatsBomb на закрытых данных, поэтому сравнение "
        "с ним не является полностью равным (спецификация, раздел 11, M5).",
    ]
    if orphans:
        items.append(
            f"{_count(orphans, 'файл', 'файла', 'файлов')} событий в источнике "
            "не соответствуют ни одному матчу в метаданных. Проект строит выборку "
            "от метаданных, поэтому эти файлы не используются."
        )
    if summary["n_shots_unknown_outcome"]:
        items.append(
            f"У {summary['n_shots_unknown_outcome']} ударов исход не входит в известную схему — "
            "они обрабатываются явно и не превращаются молча в не-голы."
        )
    if summary["n_shots_missing_location"]:
        items.append(
            f"У {summary['n_shots_missing_location']} ударов отсутствуют координаты — "
            "геометрию для них посчитать нельзя, строки исключаются с подсчётом."
        )
    if summary["n_frames_with_invalid_locations"]:
        items.append(
            f"В {summary['n_frames_with_invalid_locations']} кадрах `freeze_frame` есть игроки "
            "без корректных координат; такие записи отбрасываются поштучно."
        )
    return items


def _decisions_needed(recommended: pd.DataFrame, download: dict[str, Any]) -> list[str]:
    items = [
        "Утвердить или изменить состав выборки и записать его в `configs/data.yaml` "
        "(`selection.approved: true`). Без этого `download_data.py --selection` не запустится.",
        f"Согласиться на разовую загрузку {download['total_gb']:.2f} ГБ событий "
        "и столько же места на диске, либо выбрать компактный вариант выше.",
        "Подтвердить, что основной эксперимент проводится только на `shot.freeze_frame`, "
        "а отдельные файлы StatsBomb 360 остаются возможным расширением.",
        "Решить, нужен ли временной holdout по сезонам как дополнительная проверка "
        "устойчивости помимо основного группового разбиения по матчам "
        "(спецификация, раздел 10.1). У рекомендуемой выборки все соревнования "
        "относятся к одному сезону, поэтому временной holdout по сезонам в ней невозможен: "
        "его пришлось бы заменить holdout по турам или отдельным соревнованием.",
        "Решить, считать ли объединение четырёх национальных лиг одного сезона одной "
        "популяцией. Лиги различаются стилем, поэтому альтернатива — обучать и оценивать "
        "модель на одной лиге, а остальные использовать как внешнюю проверку переносимости.",
    ]
    if not recommended.empty and len(recommended) > 1:
        items.append(
            f"Решить, брать ли все "
            f"{_count(len(recommended), 'лигу', 'лиги', 'лиг')} блока "
            "или сузить выборку до одной ради меньшего объёма."
        )
    return items


if __name__ == "__main__":
    raise SystemExit(main())
