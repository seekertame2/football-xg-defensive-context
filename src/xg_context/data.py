"""Воспроизводимая избирательная загрузка StatsBomb Open Data.

Ключевые свойства (спецификация, раздел 5):

* источник — официальный репозиторий ``hudl/open-data``;
* ревизия зафиксирована commit SHA, все URL строятся от него;
* репозиторий не клонируется целиком: скачиваются только нужные файлы;
* повторный запуск использует локальный кеш в ``data/raw/``;
* объём загрузки оценивается заранее по git-tree, без скачивания.

Полный объём источника — около 16 ГБ (12.8 ГБ событий и 3.2 ГБ StatsBomb 360),
поэтому сплошная загрузка недопустима и в проекте не используется.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from xg_context.config import (
    RAW_DATA_DIR,
    DataConfig,
    SourceConfig,
    github_token,
)

logger = logging.getLogger(__name__)

USER_AGENT = (
    "football-xg-defensive-context/0.1 (research project; +https://github.com/hudl/open-data)"
)

COMPETITIONS_PATH = "data/competitions.json"


def matches_path(competition_id: int, season_id: int) -> str:
    """Путь к файлу матчей сезона внутри репозитория источника."""
    return f"data/matches/{competition_id}/{season_id}.json"


def events_path(match_id: int) -> str:
    """Путь к файлу событий матча внутри репозитория источника."""
    return f"data/events/{match_id}.json"


def lineups_path(match_id: int) -> str:
    """Путь к файлу составов матча внутри репозитория источника."""
    return f"data/lineups/{match_id}.json"


def three_sixty_path(match_id: int) -> str:
    """Путь к файлу StatsBomb 360 матча внутри репозитория источника."""
    return f"data/three-sixty/{match_id}.json"


@dataclass(frozen=True)
class DownloadResult:
    """Результат загрузки одного файла."""

    path: str
    local_path: Path
    from_cache: bool
    n_bytes: int


class StatsBombDownloader:
    """Загрузчик файлов StatsBomb Open Data с локальным кешем.

    Кеш повторяет структуру исходного репозитория внутри ``data/raw/``,
    поэтому путь файла однозначно соответствует пути в зафиксированной ревизии.
    """

    def __init__(
        self,
        source: SourceConfig,
        cache_dir: Path = RAW_DATA_DIR,
        timeout: int = 60,
        max_retries: int = 3,
        workers: int = 8,
    ) -> None:
        self.source = source
        self.cache_dir = Path(cache_dir)
        self.timeout = timeout
        self.max_retries = max_retries
        self.workers = workers
        self._session = requests.Session()
        headers = {"User-Agent": USER_AGENT}
        token = github_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._session.headers.update(headers)
        self.bytes_downloaded = 0
        self.files_downloaded = 0
        self.files_from_cache = 0

    # ---------------------------------------------------------------- кеш

    def local_path(self, path: str) -> Path:
        """Локальный путь кеша для файла источника.

        Кеш разделён по commit SHA, поэтому смена ревизии не смешивает данные.
        """
        return self.cache_dir / self.source.commit_sha[:12] / path.lstrip("/")

    def is_cached(self, path: str) -> bool:
        local = self.local_path(path)
        return local.exists() and local.stat().st_size > 0

    # ------------------------------------------------------------ загрузка

    def fetch(self, path: str, *, force: bool = False) -> DownloadResult:
        """Скачать один файл источника (или взять из кеша)."""
        local = self.local_path(path)
        if not force and self.is_cached(path):
            self.files_from_cache += 1
            return DownloadResult(path, local, from_cache=True, n_bytes=local.stat().st_size)

        url = self.source.raw_url(path)
        content = self.get_bytes(url)
        local.parent.mkdir(parents=True, exist_ok=True)
        tmp = local.with_suffix(local.suffix + ".part")
        tmp.write_bytes(content)
        tmp.replace(local)

        self.files_downloaded += 1
        self.bytes_downloaded += len(content)
        return DownloadResult(path, local, from_cache=False, n_bytes=len(content))

    def fetch_many(self, paths: Sequence[str], *, force: bool = False) -> list[DownloadResult]:
        """Скачать несколько файлов параллельно, сохраняя порядок входа."""
        if not paths:
            return []
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            return list(pool.map(lambda p: self.fetch(p, force=force), paths))

    def load_json(self, path: str, *, force: bool = False) -> Any:
        """Скачать (или взять из кеша) и распарсить JSON-файл источника."""
        result = self.fetch(path, force=force)
        with result.local_path.open(encoding="utf-8") as handle:
            return json.load(handle)

    def get_bytes(self, url: str) -> bytes:
        """Скачать произвольный URL источника с повторами. Используется и для GitHub API."""
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._session.get(url, timeout=self.timeout)
                if response.status_code == 404:
                    raise FileNotFoundError(f"Файл отсутствует в источнике: {url}")
                response.raise_for_status()
                return response.content
            except FileNotFoundError:
                raise
            except Exception as error:
                last_error = error
                if attempt < self.max_retries:
                    delay = 2.0 * attempt
                    logger.warning(
                        "Ошибка загрузки %s (попытка %d/%d): %s. Повтор через %.0f с.",
                        url,
                        attempt,
                        self.max_retries,
                        error,
                        delay,
                    )
                    time.sleep(delay)
        raise RuntimeError(
            f"Не удалось скачать {url} за {self.max_retries} попыток"
        ) from last_error

    # ------------------------------------------------------- удобные обёртки

    def load_competitions(self) -> list[dict[str, Any]]:
        """Прочитать `competitions.json` — перечень соревнований и сезонов."""
        return self.load_json(COMPETITIONS_PATH)

    def load_matches(self, competition_id: int, season_id: int) -> list[dict[str, Any]]:
        """Прочитать список матчей одного competition-season."""
        return self.load_json(matches_path(competition_id, season_id))

    def load_events(self, match_id: int) -> list[dict[str, Any]]:
        """Прочитать все события матча."""
        return self.load_json(events_path(match_id))

    def load_three_sixty(self, match_id: int) -> list[dict[str, Any]]:
        """Прочитать кадры StatsBomb 360 матча."""
        return self.load_json(three_sixty_path(match_id))


# --------------------------------------------------------------------------------------
# Инвентаризация источника через git-tree
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceInventory:
    """Полный перечень файлов зафиксированной ревизии с их размерами.

    Один запрос к GitHub API возвращает всё дерево, что позволяет:

    * точно узнать, для каких матчей существуют события и файлы 360;
    * оценить объём будущей загрузки, ничего не скачивая.
    """

    commit_sha: str
    sizes: dict[str, int]

    @property
    def paths(self) -> frozenset[str]:
        return frozenset(self.sizes)

    def size_of(self, path: str) -> int:
        """Размер файла источника в байтах (0, если файла нет)."""
        return self.sizes.get(path, 0)

    def total_size(self, paths: Iterable[str]) -> int:
        """Суммарный размер набора файлов в байтах."""
        return sum(self.size_of(p) for p in paths)

    def has_events(self, match_id: int) -> bool:
        return events_path(match_id) in self.sizes

    def has_three_sixty(self, match_id: int) -> bool:
        return three_sixty_path(match_id) in self.sizes

    def match_ids_with_events(self) -> set[int]:
        return self._match_ids_under("data/events/")

    def match_ids_with_three_sixty(self) -> set[int]:
        return self._match_ids_under("data/three-sixty/")

    def _match_ids_under(self, prefix: str) -> set[int]:
        out: set[int] = set()
        for path in self.sizes:
            if path.startswith(prefix) and path.endswith(".json"):
                stem = path[len(prefix) : -len(".json")]
                if stem.isdigit():
                    out.add(int(stem))
        return out


def fetch_source_inventory(
    downloader: StatsBombDownloader,
    *,
    force: bool = False,
) -> SourceInventory:
    """Получить инвентаризацию источника, кешируя ответ git-tree на диск."""
    cache_file = downloader.local_path("_meta/git_tree.json")
    if not force and cache_file.exists() and cache_file.stat().st_size > 0:
        logger.info("git-tree взят из кеша: %s", cache_file)
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
    else:
        url = downloader.source.tree_url()
        logger.info("Запрашиваю git-tree ревизии %s", downloader.source.commit_sha[:12])
        payload = json.loads(downloader.get_bytes(url).decode("utf-8"))
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(payload), encoding="utf-8")

    if payload.get("truncated"):
        raise RuntimeError(
            "GitHub вернул усечённое дерево репозитория — инвентаризация неполна. "
            "Повторите запрос или используйте постраничный обход."
        )

    sizes = {
        entry["path"]: int(entry.get("size", 0))
        for entry in payload["tree"]
        if entry.get("type") == "blob"
    }
    return SourceInventory(commit_sha=downloader.source.commit_sha, sizes=sizes)


# --------------------------------------------------------------------------------------
# Планирование загрузки
# --------------------------------------------------------------------------------------


def estimate_download_size(
    inventory: SourceInventory,
    match_ids: Iterable[int],
    *,
    include_events: bool = True,
    include_three_sixty: bool = False,
    include_lineups: bool = False,
) -> dict[str, Any]:
    """Оценить объём загрузки для набора матчей, ничего не скачивая."""
    match_ids = list(match_ids)
    paths: list[str] = []
    if include_events:
        paths += [events_path(m) for m in match_ids if inventory.has_events(m)]
    if include_lineups:
        paths += [lineups_path(m) for m in match_ids if lineups_path(m) in inventory.sizes]
    if include_three_sixty:
        paths += [three_sixty_path(m) for m in match_ids if inventory.has_three_sixty(m)]

    total_bytes = inventory.total_size(paths)
    return {
        "n_matches": len(match_ids),
        "n_files": len(paths),
        "total_bytes": total_bytes,
        "total_mb": round(total_bytes / 1e6, 1),
        "total_gb": round(total_bytes / 1e9, 3),
    }


def download_match_files(
    downloader: StatsBombDownloader,
    match_ids: Sequence[int],
    *,
    inventory: SourceInventory | None = None,
    include_events: bool = True,
    include_three_sixty: bool = False,
    include_lineups: bool = False,
) -> list[DownloadResult]:
    """Скачать выбранные файлы матчей, пропуская отсутствующие в источнике."""
    paths: list[str] = []
    for match_id in match_ids:
        if include_events and (inventory is None or inventory.has_events(match_id)):
            paths.append(events_path(match_id))
        if include_lineups:
            paths.append(lineups_path(match_id))
        if include_three_sixty and (inventory is None or inventory.has_three_sixty(match_id)):
            paths.append(three_sixty_path(match_id))

    logger.info("Загружаю %d файлов (%d матчей)", len(paths), len(match_ids))
    return downloader.fetch_many(paths)


def download_all_metadata(
    downloader: StatsBombDownloader,
) -> tuple[list[dict[str, Any]], dict[tuple[int, int], list[dict[str, Any]]]]:
    """Скачать `competitions.json` и все файлы матчей (около 7 МБ суммарно).

    Метаданные малы, поэтому берутся целиком: это даёт полную перепись
    соревнований, сезонов и матчей без единой догадки.
    """
    competitions = downloader.load_competitions()
    keys = [(int(c["competition_id"]), int(c["season_id"])) for c in competitions]
    downloader.fetch_many([matches_path(c, s) for c, s in keys])

    matches_by_season: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for competition_id, season_id in keys:
        matches_by_season[(competition_id, season_id)] = downloader.load_matches(
            competition_id, season_id
        )
    return competitions, matches_by_season


def build_downloader(config: DataConfig) -> StatsBombDownloader:
    """Создать загрузчик по конфигурации проекта."""
    return StatsBombDownloader(
        source=config.source,
        timeout=config.request_timeout,
        max_retries=config.max_retries,
        workers=config.download_workers,
    )
