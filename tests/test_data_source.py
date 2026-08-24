"""Тесты загрузчика и инвентаризации источника.

Сеть здесь не используется: проверяются построение URL от зафиксированного
commit SHA, раскладка кеша, разбор git-tree и оценка объёма загрузки.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xg_context.config import SourceConfig, load_data_config
from xg_context.data import (
    SourceInventory,
    StatsBombDownloader,
    estimate_download_size,
    events_path,
    lineups_path,
    matches_path,
    three_sixty_path,
)

SHA = "b0bc9f22dd77c206ddedc1d742893b3bbe64baec"


@pytest.fixture
def source() -> SourceConfig:
    return SourceConfig(
        repo="hudl/open-data",
        commit_sha=SHA,
        raw_base_url="https://raw.githubusercontent.com",
        api_base_url="https://api.github.com",
    )


@pytest.fixture
def inventory() -> SourceInventory:
    return SourceInventory(
        commit_sha=SHA,
        sizes={
            "data/competitions.json": 5_000,
            "data/matches/43/3.json": 100_000,
            "data/events/8650.json": 3_000_000,
            "data/events/8651.json": 2_000_000,
            "data/lineups/8650.json": 20_000,
            "data/three-sixty/8650.json": 7_500_000,
        },
    )


class TestSourcePaths:
    def test_paths_follow_statsbomb_layout(self) -> None:
        assert matches_path(43, 3) == "data/matches/43/3.json"
        assert events_path(8650) == "data/events/8650.json"
        assert lineups_path(8650) == "data/lineups/8650.json"
        assert three_sixty_path(8650) == "data/three-sixty/8650.json"

    def test_raw_url_is_pinned_to_the_commit(self, source: SourceConfig) -> None:
        url = source.raw_url("data/competitions.json")
        assert url == (
            f"https://raw.githubusercontent.com/hudl/open-data/{SHA}/data/competitions.json"
        )
        assert SHA in url, "URL обязан быть привязан к зафиксированной ревизии, а не к ветке"

    def test_raw_url_tolerates_leading_slash(self, source: SourceConfig) -> None:
        assert source.raw_url("/data/competitions.json") == source.raw_url("data/competitions.json")

    def test_tree_url_targets_the_same_commit(self, source: SourceConfig) -> None:
        assert source.tree_url().endswith(f"/git/trees/{SHA}?recursive=1")

    def test_permalink_points_to_the_revision(self, source: SourceConfig) -> None:
        assert source.permalink == f"https://github.com/hudl/open-data/tree/{SHA}"


class TestCacheLayout:
    def test_cache_is_partitioned_by_commit(self, source: SourceConfig, tmp_path: Path) -> None:
        """Смена ревизии не должна смешивать данные в одном кеше."""
        downloader = StatsBombDownloader(source, cache_dir=tmp_path)
        local = downloader.local_path("data/events/8650.json")
        assert SHA[:12] in local.parts
        assert local.name == "8650.json"

    def test_missing_file_is_not_cached(self, source: SourceConfig, tmp_path: Path) -> None:
        downloader = StatsBombDownloader(source, cache_dir=tmp_path)
        assert downloader.is_cached("data/events/8650.json") is False

    def test_existing_file_is_detected(self, source: SourceConfig, tmp_path: Path) -> None:
        downloader = StatsBombDownloader(source, cache_dir=tmp_path)
        local = downloader.local_path("data/events/8650.json")
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text("[]", encoding="utf-8")
        assert downloader.is_cached("data/events/8650.json") is True

    def test_empty_file_is_not_treated_as_cached(
        self, source: SourceConfig, tmp_path: Path
    ) -> None:
        """Обрывок загрузки не должен считаться валидным кешем."""
        downloader = StatsBombDownloader(source, cache_dir=tmp_path)
        local = downloader.local_path("data/events/8650.json")
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(b"")
        assert downloader.is_cached("data/events/8650.json") is False

    def test_cached_file_is_read_without_network(
        self, source: SourceConfig, tmp_path: Path
    ) -> None:
        downloader = StatsBombDownloader(source, cache_dir=tmp_path)
        local = downloader.local_path("data/competitions.json")
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text(json.dumps([{"competition_id": 43}]), encoding="utf-8")
        assert downloader.load_json("data/competitions.json") == [{"competition_id": 43}]
        assert downloader.files_downloaded == 0
        assert downloader.files_from_cache == 1


class TestInventory:
    def test_detects_available_files(self, inventory: SourceInventory) -> None:
        assert inventory.has_events(8650) is True
        assert inventory.has_events(999999) is False
        assert inventory.has_three_sixty(8650) is True
        assert inventory.has_three_sixty(8651) is False

    def test_lists_match_ids(self, inventory: SourceInventory) -> None:
        assert inventory.match_ids_with_events() == {8650, 8651}
        assert inventory.match_ids_with_three_sixty() == {8650}

    def test_size_of_missing_file_is_zero(self, inventory: SourceInventory) -> None:
        assert inventory.size_of("data/events/999999.json") == 0

    def test_total_size_sums_only_known_files(self, inventory: SourceInventory) -> None:
        total = inventory.total_size(["data/events/8650.json", "data/events/999999.json"])
        assert total == 3_000_000


class TestDownloadEstimate:
    def test_events_only_estimate(self, inventory: SourceInventory) -> None:
        estimate = estimate_download_size(inventory, [8650, 8651])
        assert estimate["n_matches"] == 2
        assert estimate["n_files"] == 2
        assert estimate["total_bytes"] == 5_000_000
        assert estimate["total_mb"] == pytest.approx(5.0)

    def test_three_sixty_adds_volume(self, inventory: SourceInventory) -> None:
        """Файлы 360 существенно дороже по объёму — оценка обязана это показывать."""
        without = estimate_download_size(inventory, [8650])
        with_360 = estimate_download_size(inventory, [8650], include_three_sixty=True)
        assert with_360["total_bytes"] > without["total_bytes"]
        assert with_360["n_files"] == without["n_files"] + 1

    def test_missing_matches_are_skipped(self, inventory: SourceInventory) -> None:
        estimate = estimate_download_size(inventory, [8650, 999999])
        assert estimate["n_matches"] == 2
        assert estimate["n_files"] == 1

    def test_empty_selection(self, inventory: SourceInventory) -> None:
        estimate = estimate_download_size(inventory, [])
        assert estimate["n_files"] == 0
        assert estimate["total_bytes"] == 0


class TestProjectDataConfig:
    def test_shipped_config_pins_a_full_commit_sha(self) -> None:
        """configs/data.yaml обязан фиксировать ревизию, а не ветку."""
        config = load_data_config()
        assert len(config.source.commit_sha) == 40
        assert config.source.commit_sha.isalnum()
        assert config.source.repo == "hudl/open-data"

    def test_selection_is_approved_and_well_formed(self) -> None:
        """Выборка утверждена человеком после аудита.

        До утверждения `approved` был False и загрузка отказывалась работать;
        после утверждения контракт другой — состав должен быть непустым
        и каждый элемент должен содержать оба идентификатора.
        """
        config = load_data_config()
        selection = config.selection
        assert selection.get("approved") is True
        pairs = selection.get("competition_seasons") or []
        assert pairs, "Утверждённая выборка не может быть пустой"
        for item in pairs:
            assert isinstance(item["competition_id"], int)
            assert isinstance(item["season_id"], int)
        assert selection.get("include_three_sixty") is False, (
            "StatsBomb 360 не используется: ни один матч выборки не имеет файла 360"
        )

    def test_selection_matches_the_approved_four_leagues(self) -> None:
        """Регрессия на состав: четыре мужские топ-лиги сезона 2015/2016."""
        config = load_data_config()
        pairs = {
            (int(i["competition_id"]), int(i["season_id"]))
            for i in config.selection["competition_seasons"]
        }
        assert pairs == {(11, 27), (2, 27), (12, 27), (7, 27)}
