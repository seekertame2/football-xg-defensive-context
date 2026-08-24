"""Константы проекта: геометрия поля, схема StatsBomb, пути и списки колонок."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

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

# Система координат StatsBomb: поле 120 x 80 единиц.
# Атака всегда идёт в сторону x = 120.
# Ворота соперника лежат на линии x = 120 между y = 36 и y = 44.
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

# Радиусы для подсчёта соперников вокруг бьющего, в единицах координат StatsBomb.
OPPONENT_RADII: tuple[float, ...] = (1.0, 2.0, 3.0, 5.0)

# Допуск для численно неустойчивых геометрических случаев.
GEOMETRY_EPS: float = 1e-9

SHOT_EVENT_TYPE: str = "Shot"
GOALKEEPER_POSITION_NAME: str = "Goalkeeper"

# Исход удара, который считается голом.
GOAL_OUTCOME_NAME: str = "Goal"

# Исходы удара в схеме StatsBomb.
# Написание сверено с данными.
# Провайдер пишет "Saved Off Target" и "Saved to Post" со строчной "to".
# Исход вне этого набора поднимает ошибку и не превращается молча в не-гол.
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

# Пенальти исключаются: это отдельный стандартизированный процесс.
PENALTY_SHOT_TYPE: str = "Penalty"
EXCLUDED_SHOT_TYPES: frozenset[str] = frozenset({PENALTY_SHOT_TYPE})

# Период серии пенальти в схеме StatsBomb.
SHOOTOUT_PERIOD: int = 5

# Списки колонок это контракт проекта.
# Пайплайн падает, если запрещённое поле попало в матрицу признаков (см.
# `features.assert_no_forbidden_columns`).
TARGET_COLUMN: str = "is_goal"

# Служебные поля.
# Хранятся в датасете для разбиения и аналитики, но никогда не попадают в матрицу признаков.
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

# Поля, которые нельзя передавать в модель ни при каких условиях.
# Либо раскрывают исход удара, либо относятся к идентичности игрока/команды/лиги.
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
    # Идентичность игрока, команды и лиги.
    # Лига нужна для разбиения и аналитики, но признаком модели не является.
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

# Четыре группы признаков для ablation.
# Разделение принципиально.
# Нужно изолировать вклад самостоятельно рассчитанного контекста.
# Смешивать его с готовыми флагами StatsBomb нельзя.

# Уровень 1: чистая геометрия удара.
GEOMETRY_FEATURES: tuple[str, ...] = (
    "shot_distance",
    "shot_angle",
)

# Уровень 2: чем, как и из какой ситуации бьют.
SHOT_CHARACTERISTIC_FEATURES: tuple[str, ...] = (
    "body_part",
    "shot_technique",
    "shot_type",
    "play_pattern",
    "is_first_time",
)

# Уровень 3: готовые флаги StatsBomb.
# Это тоже контекст, но не наш расчёт, а разметка провайдера.
# Держим отдельно.
# Иначе прирост от собственной геометрии был бы завышен за счёт чужой работы.
STATSBOMB_CONTEXT_FLAG_FEATURES: tuple[str, ...] = (
    "under_pressure",
    "one_on_one",
    "open_goal",
)

# Уровень 4: пространственный контекст обороны, рассчитанный самостоятельно.
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

BASELINE_FEATURES: tuple[str, ...] = GEOMETRY_FEATURES

SHOT_CONTEXT_FEATURES: tuple[str, ...] = (
    *SHOT_CHARACTERISTIC_FEATURES,
    *STATSBOMB_CONTEXT_FLAG_FEATURES,
)

# Категориальные признаки (остальные считаются числовыми).
CATEGORICAL_FEATURES: frozenset[str] = frozenset(
    {"body_part", "shot_technique", "shot_type", "play_pattern"}
)

# Наборы признаков для лестницы моделей и ablation.
# Ключ - имя набора, значение - список колонок.
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
    # Sensitivity test: L4 без `n_opponents_visible`.
    # Признак отражает и плотность обороны, и границы поля зрения камеры.
    # Поэтому эффект проверяется ещё и без него.
    "geometry_shot_flags_defensive_no_visible": (
        *GEOMETRY_FEATURES,
        *SHOT_CHARACTERISTIC_FEATURES,
        *STATSBOMB_CONTEXT_FLAG_FEATURES,
        *tuple(f for f in DEFENSIVE_CONTEXT_FEATURES if f != "n_opponents_visible"),
    ),
}

# Признак, вокруг которого построен sensitivity test.
AMBIGUOUS_VISIBILITY_FEATURE: str = "n_opponents_visible"

# Поля StatsBomb, которые есть в JSON только когда равны true.
# Отсутствие такого поля трактуется как false только после проверки на данных.
# Проверяется, что значение false в данных не встречается.
# Проверка живёт в `dataset.py`.
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

RANDOM_SEED: int = 42

# Версия конфигурации построения датасета.
# Попадает в data_manifest.json.
DATASET_CONFIG_VERSION: str = "1.0.0"


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
    """Параметры выборочного аудита данных."""

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
    """Опциональный токен GitHub. Проект зовёт API один раз, токен не обязателен."""
    return os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
