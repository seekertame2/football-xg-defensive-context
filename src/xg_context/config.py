"""Централизованная конфигурация проекта.

Здесь собраны все константы, которые иначе расползлись бы по ноутбукам:
геометрия поля, система координат StatsBomb, маппинг target,
пути проекта и явные списки колонок (спецификация, разделы 9 и 18).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# --------------------------------------------------------------------------------------
# Пути проекта
# --------------------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
INTERIM_DATA_DIR: Path = DATA_DIR / "interim"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
DATA_MANIFEST_PATH: Path = DATA_DIR / "data_manifest.json"

CONFIGS_DIR: Path = PROJECT_ROOT / "configs"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
TABLES_DIR: Path = REPORTS_DIR / "tables"

# --------------------------------------------------------------------------------------
# Система координат StatsBomb
#
# Поле 120 x 80 условных ярдов. Атака всегда идёт в сторону x = 120,
# ворота соперника расположены на линии x = 120 между y = 36 и y = 44.
# Точка (0, 0) — левый верхний угол в терминах StatsBomb.
# --------------------------------------------------------------------------------------

PITCH_LENGTH: float = 120.0
PITCH_WIDTH: float = 80.0

GOAL_LINE_X: float = 120.0
GOAL_WIDTH: float = 8.0
GOAL_CENTER_Y: float = PITCH_WIDTH / 2.0  # 40.0
GOAL_LEFT_POST_Y: float = GOAL_CENTER_Y - GOAL_WIDTH / 2.0  # 36.0
GOAL_RIGHT_POST_Y: float = GOAL_CENTER_Y + GOAL_WIDTH / 2.0  # 44.0

GOAL_CENTER: tuple[float, float] = (GOAL_LINE_X, GOAL_CENTER_Y)
LEFT_POST: tuple[float, float] = (GOAL_LINE_X, GOAL_LEFT_POST_Y)
RIGHT_POST: tuple[float, float] = (GOAL_LINE_X, GOAL_RIGHT_POST_Y)

#: Радиусы (в ярдах) для подсчёта числа соперников вокруг бьющего.
OPPONENT_RADII: tuple[float, ...] = (1.0, 2.0, 3.0, 5.0)

#: Допуск для численно неустойчивых геометрических случаев.
GEOMETRY_EPS: float = 1e-9

# --------------------------------------------------------------------------------------
# Схема StatsBomb: типы событий и исходы ударов
# --------------------------------------------------------------------------------------

SHOT_EVENT_TYPE: str = "Shot"
GOALKEEPER_POSITION_NAME: str = "Goalkeeper"

#: Исход удара, который считается голом.
GOAL_OUTCOME_NAME: str = "Goal"

#: Полный набор исходов удара в схеме StatsBomb Open Data (Open Data Events v4.0.0).
#: Написание сверено с фактическими данными зафиксированной ревизии на выборке
#: аудита: StatsBomb использует именно "Saved Off Target" и "Saved to Post"
#: (со строчной "to"), а не варианты "Saved Off T" / "Saved To Post".
#: Исходы вне этого множества обрабатываются явно и не превращаются молча в 0.
KNOWN_SHOT_OUTCOMES: frozenset[str] = frozenset(
    {
        "Goal",
        "Saved",
        "Blocked",
        "Off T",
        "Post",
        "Wayward",
        "Saved Off Target",
        "Saved to Post",
    }
)

#: Типы ударов, которые исключаются из основной выборки (спецификация, раздел 8.3).
PENALTY_SHOT_TYPE: str = "Penalty"
EXCLUDED_SHOT_TYPES: frozenset[str] = frozenset({PENALTY_SHOT_TYPE})

#: Период серии пенальти в схеме StatsBomb.
SHOOTOUT_PERIOD: int = 5

# --------------------------------------------------------------------------------------
# Явные списки колонок (спецификация, раздел 9)
#
# Эти списки — контракт проекта. Пайплайн обязан падать, если запрещённое поле
# попадает в матрицу признаков (см. `xg_context.features.assert_no_forbidden_columns`).
# --------------------------------------------------------------------------------------

TARGET_COLUMN: str = "is_goal"

#: Служебные поля. Хранятся в датасете для разбиения и аналитики,
#: но никогда не попадают в матрицу признаков.
ID_COLUMNS: tuple[str, ...] = (
    "shot_id",
    "match_id",
    "competition_id",
    "season_id",
    "competition_name",
    "match_date",
    "period",
    "minute",
    "second",
    "player_id",
    "player_name",
    "team_id",
    "team_name",
)

BENCHMARK_COLUMNS: tuple[str, ...] = ("statsbomb_xg",)

#: Поля, которые нельзя передавать в модель ни при каких условиях.
#: Либо раскрывают исход удара, либо относятся к идентичности игрока/команды/лиги.
FORBIDDEN_FEATURE_COLUMNS: tuple[str, ...] = (
    # benchmark
    "statsbomb_xg",
    # исход удара и производные
    "is_goal",
    "shot_outcome",
    "shot_outcome_id",
    "outcome",
    "goal",
    # траектория после удара
    "end_location",
    "shot_end_location",
    "shot_end_x",
    "shot_end_y",
    "shot_end_z",
    # поля, раскрывающие результат
    "shot_saved_off_target",
    "shot_saved_to_post",
    "shot_deflected",
    "goalkeeper_outcome",
    "goalkeeper_technique",
    "block",
    "save",
    # идентичность игрока, команды и лиги
    # (решение владельца проекта: лига используется для разбиения и аналитики,
    #  но не как признак основной модели)
    "player_id",
    "player_name",
    "team_id",
    "team_name",
    "possession_team_id",
    "possession_team_name",
    "competition_id",
    "competition_name",
    "season_id",
    # итоговый счёт и служебные идентификаторы
    "home_score",
    "away_score",
    "match_id",
    "shot_id",
)

# --------------------------------------------------------------------------------------
# Четыре группы признаков ablation study.
#
# Разделение принципиально: спецификация требует изолировать вклад
# САМОСТОЯТЕЛЬНО РАССЧИТАННОГО пространственного контекста, а не смешивать его
# с готовыми флагами StatsBomb, которые тоже описывают обстановку вокруг удара.
# --------------------------------------------------------------------------------------

#: Уровень 1 — чистая геометрия удара (M1).
GEOMETRY_FEATURES: tuple[str, ...] = (
    "shot_distance",
    "shot_angle",
)

#: Уровень 2 — внутренние характеристики удара: чем, как и из какой ситуации бьют.
SHOT_CHARACTERISTIC_FEATURES: tuple[str, ...] = (
    "body_part",
    "shot_technique",
    "shot_type",
    "play_pattern",
    "is_first_time",
)

#: Уровень 3 — готовые контекстные флаги StatsBomb.
#: Это тоже контекст, но НЕ рассчитанный нами: он приходит из разметки провайдера.
#: Держим его отдельно, иначе прирост от собственной геометрии защитников
#: был бы завышен за счёт чужой работы.
STATSBOMB_CONTEXT_FLAG_FEATURES: tuple[str, ...] = (
    "under_pressure",
    "one_on_one",
    "open_goal",
)

#: Уровень 4 — самостоятельно рассчитанный пространственный контекст обороны (M4).
#: Центральный предмет исследования.
DEFENSIVE_CONTEXT_FEATURES: tuple[str, ...] = (
    "nearest_opponent_distance",
    "opponents_within_1y",
    "opponents_within_2y",
    "opponents_within_3y",
    "opponents_within_5y",
    "opponents_in_shot_cone",
    "opponents_between_shot_and_goal",
    "n_opponents_visible",
    "goalkeeper_distance_to_shot",
    "goalkeeper_distance_to_goal_line",
    "goalkeeper_lateral_offset",
    "goalkeeper_distance_to_shot_line",
    "goalkeeper_in_shot_cone",
    "has_goalkeeper",
)

#: Совместимость: M1 использует только геометрию.
BASELINE_FEATURES: tuple[str, ...] = GEOMETRY_FEATURES

#: M2 — характеристики удара плюс готовые флаги StatsBomb.
SHOT_CONTEXT_FEATURES: tuple[str, ...] = (
    *SHOT_CHARACTERISTIC_FEATURES,
    *STATSBOMB_CONTEXT_FLAG_FEATURES,
)

#: Категориальные признаки (остальные считаются числовыми).
CATEGORICAL_FEATURES: frozenset[str] = frozenset(
    {"body_part", "shot_technique", "shot_type", "play_pattern"}
)

#: Наборы признаков для лестницы моделей и ablation.
#: Ключ — имя набора, значение — список колонок.
FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "geometry": GEOMETRY_FEATURES,
    "geometry_shot": (*GEOMETRY_FEATURES, *SHOT_CHARACTERISTIC_FEATURES),
    "geometry_shot_flags": (
        *GEOMETRY_FEATURES,
        *SHOT_CHARACTERISTIC_FEATURES,
        *STATSBOMB_CONTEXT_FLAG_FEATURES,
    ),
    "geometry_shot_flags_defensive": (
        *GEOMETRY_FEATURES,
        *SHOT_CHARACTERISTIC_FEATURES,
        *STATSBOMB_CONTEXT_FLAG_FEATURES,
        *DEFENSIVE_CONTEXT_FEATURES,
    ),
    "geometry_defensive": (*GEOMETRY_FEATURES, *DEFENSIVE_CONTEXT_FEATURES),
}

#: Флаги StatsBomb, которые присутствуют в JSON только когда равны true.
#: Отсутствие такого поля трактуется как false — но только после явной проверки
#: на реальных данных, что поле никогда не принимает значение false
#: (спецификация, раздел 9.5). Проверка живёт в `xg_context.dataset`.
SPARSE_BOOLEAN_FIELDS: tuple[str, ...] = (
    "under_pressure",
    "one_on_one",
    "open_goal",
    "first_time",
    "aerial_won",
    "follows_dribble",
    "redirect",
    "deflected",
)

# --------------------------------------------------------------------------------------
# Воспроизводимость
# --------------------------------------------------------------------------------------

RANDOM_SEED: int = 42

#: Версия конфигурации построения датасета. Попадает в data_manifest.json.
DATASET_CONFIG_VERSION: str = "0.1.0"


# --------------------------------------------------------------------------------------
# Конфигурация загрузки данных (configs/data.yaml)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceConfig:
    """Неизменяемая ревизия источника StatsBomb Open Data."""

    repo: str
    commit_sha: str
    raw_base_url: str
    api_base_url: str

    def raw_url(self, path: str) -> str:
        """Собрать URL файла в зафиксированной ревизии источника."""
        return f"{self.raw_base_url}/{self.repo}/{self.commit_sha}/{path.lstrip('/')}"

    def tree_url(self) -> str:
        """URL git-tree всей ревизии (одним запросом даёт список файлов и размеры)."""
        return f"{self.api_base_url}/repos/{self.repo}/git/trees/{self.commit_sha}?recursive=1"

    @property
    def source_url(self) -> str:
        return f"https://github.com/{self.repo}"

    @property
    def permalink(self) -> str:
        return f"https://github.com/{self.repo}/tree/{self.commit_sha}"


@dataclass(frozen=True)
class AuditConfig:
    """Параметры аудита данных (этап 1)."""

    matches_per_season: int
    max_event_files: int
    sample_seed: int
    three_sixty_probe_matches: int


@dataclass(frozen=True)
class DataConfig:
    """Полная конфигурация загрузки и аудита."""

    source: SourceConfig
    audit: AuditConfig
    download_workers: int
    request_timeout: int
    max_retries: int
    selection: dict[str, Any] = field(default_factory=dict)


def load_data_config(path: str | Path | None = None) -> DataConfig:
    """Прочитать `configs/data.yaml` и вернуть типизированную конфигурацию."""
    config_path = Path(path) if path is not None else CONFIGS_DIR / "data.yaml"
    with config_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    source = SourceConfig(**payload["source"])
    audit = AuditConfig(**payload["audit"])
    return DataConfig(
        source=source,
        audit=audit,
        download_workers=int(payload.get("download_workers", 8)),
        request_timeout=int(payload.get("request_timeout", 60)),
        max_retries=int(payload.get("max_retries", 3)),
        selection=payload.get("selection") or {},
    )


def github_token() -> str | None:
    """Опциональный токен GitHub — поднимает лимит GitHub API с 60 до 5000 запросов в час.

    Проект использует API ровно один раз (git tree), поэтому токен не обязателен.
    """
    return os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
