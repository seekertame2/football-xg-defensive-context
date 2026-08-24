"""Тесты группового разбиения выборки.

Главное правило: удары одного матча не могут попадать в разные части выборки.
Дополнительно проверяются детерминированность, сопоставимые доли лиг и сопоставимая доля голов.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from xg_context.splitting import (
    SPLIT_NAMES,
    load_split,
    make_grouped_split,
    save_split,
    split_summary,
)

LEAGUES = ("La Liga", "Premier League", "Serie A", "Ligue 1")


def make_shots(
    n_matches_per_league: int = 60,
    shots_per_match: int = 24,
    goal_rate: float = 0.10,
    seed: int = 7,
) -> pd.DataFrame:
    """Синтетическая таблица ударов с реалистичной структурой матчей и лиг."""
    rng = np.random.default_rng(seed)
    rows = []
    match_id = 1000
    for league in LEAGUES:
        for _ in range(n_matches_per_league):
            match_id += 1
            n_shots = int(rng.integers(shots_per_match - 8, shots_per_match + 9))
            for index in range(n_shots):
                rows.append(
                    {
                        "shot_id": f"{match_id}-{index}",
                        "match_id": match_id,
                        "competition_name": league,
                        "is_goal": int(rng.random() < goal_rate),
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def shots() -> pd.DataFrame:
    return make_shots()


@pytest.fixture(scope="module")
def split(shots: pd.DataFrame):
    return make_grouped_split(shots, seed=42)


class TestNoLeakageBetweenSplits:
    """Центральное требование: матч целиком принадлежит одной части."""

    def test_match_ids_do_not_overlap(self, split) -> None:
        sets = {name: set(split.match_ids(name)) for name in SPLIT_NAMES}
        assert sets["train"] & sets["validation"] == set()
        assert sets["train"] & sets["test"] == set()
        assert sets["validation"] & sets["test"] == set()

    def test_every_match_is_assigned_exactly_once(self, shots, split) -> None:
        assigned = sum(len(split.match_ids(name)) for name in SPLIT_NAMES)
        assert assigned == shots["match_id"].nunique()

    def test_shot_ids_do_not_overlap(self, shots, split) -> None:
        frame = shots.copy()
        frame["split"] = split.series(frame["match_id"])
        groups = {name: set(frame.loc[frame["split"] == name, "shot_id"]) for name in SPLIT_NAMES}
        assert groups["train"] & groups["validation"] == set()
        assert groups["train"] & groups["test"] == set()
        assert groups["validation"] & groups["test"] == set()

    def test_all_shots_of_a_match_share_one_split(self, shots, split) -> None:
        frame = shots.copy()
        frame["split"] = split.series(frame["match_id"])
        pairs = frame.drop_duplicates(subset=["match_id", "split"])
        assert len(pairs) == frame["match_id"].nunique(), "Матч разъехался между частями выборки"

    def test_no_shot_is_left_unassigned(self, shots, split) -> None:
        frame = shots.copy()
        frame["split"] = split.series(frame["match_id"])
        assert frame["split"].notna().all()


class TestSizes:
    def test_proportions_are_close_to_target(self, shots, split) -> None:
        summary = split_summary(shots, split)
        shares = dict(zip(summary["split"], summary["share_of_shots"], strict=True))
        assert shares["train"] == pytest.approx(0.70, abs=0.02)
        assert shares["validation"] == pytest.approx(0.15, abs=0.02)
        assert shares["test"] == pytest.approx(0.15, abs=0.02)

    def test_rejects_sizes_that_do_not_sum_to_one(self, shots) -> None:
        with pytest.raises(ValueError, match="суммироваться"):
            make_grouped_split(shots, train_size=0.7, validation_size=0.2, test_size=0.2)

    def test_rejects_empty_frame(self) -> None:
        with pytest.raises(ValueError, match="Пустая таблица"):
            make_grouped_split(pd.DataFrame(columns=["match_id", "is_goal", "competition_name"]))


class TestBalance:
    """Доли лиг и доля голов должны быть сопоставимы между частями."""

    def test_league_shares_are_close_across_splits(self, shots, split) -> None:
        summary = split_summary(shots, split)
        for league in LEAGUES:
            column = f"share_{league}"
            values = summary[column].to_numpy()
            assert values.max() - values.min() < 0.03, (
                f"Доли лиги {league} по частям выборки расходятся: {values}"
            )

    def test_every_league_is_present_in_every_split(self, shots, split) -> None:
        frame = shots.copy()
        frame["split"] = split.series(frame["match_id"])
        for name in SPLIT_NAMES:
            present = set(frame.loc[frame["split"] == name, "competition_name"])
            assert present == set(LEAGUES)

    def test_goal_rates_are_close_across_splits(self, shots, split) -> None:
        summary = split_summary(shots, split)
        rates = summary["goal_rate"].to_numpy()
        assert rates.max() - rates.min() < 0.02, f"Доли голов расходятся: {rates}"

    def test_every_split_contains_goals(self, shots, split) -> None:
        summary = split_summary(shots, split)
        assert (summary["n_goals"] > 0).all()


class TestDeterminism:
    def test_same_seed_gives_same_split(self, shots) -> None:
        first = make_grouped_split(shots, seed=42)
        second = make_grouped_split(shots, seed=42)
        assert first.assignment == second.assignment

    def test_different_seed_gives_different_split(self, shots) -> None:
        first = make_grouped_split(shots, seed=42)
        second = make_grouped_split(shots, seed=2024)
        assert first.assignment != second.assignment

    def test_row_order_does_not_change_the_split(self, shots) -> None:
        """Разбиение зависит от данных, а не от порядка строк в файле."""
        shuffled = shots.sample(frac=1.0, random_state=1).reset_index(drop=True)
        assert make_grouped_split(shots, seed=42).assignment == (
            make_grouped_split(shuffled, seed=42).assignment
        )


class TestPersistence:
    def test_saved_split_round_trips(self, shots, split, tmp_path) -> None:
        frame = shots.copy()
        frame["split"] = split.series(frame["match_id"])
        shot_ids = {
            name: frame.loc[frame["split"] == name, "shot_id"].tolist() for name in SPLIT_NAMES
        }
        path = tmp_path / "split.json"
        save_split(split, path, shot_ids)

        loaded = load_split(path)
        assert loaded["seed"] == split.seed
        assert loaded["match_ids"]["test"] == split.match_ids("test")
        assert loaded["shot_ids"]["test"] == shot_ids["test"]

    def test_saved_file_is_valid_json(self, split, tmp_path) -> None:
        path = tmp_path / "split.json"
        save_split(split, path, {name: [] for name in SPLIT_NAMES})
        json.loads(path.read_text(encoding="utf-8"))


class TestWithoutStratification:
    def test_works_when_no_stratify_column_given(self, shots) -> None:
        split = make_grouped_split(shots, stratify_column=None, seed=42)
        frame = shots.copy()
        frame["split"] = split.series(frame["match_id"])
        pairs = frame.drop_duplicates(subset=["match_id", "split"])
        assert len(pairs) == frame["match_id"].nunique()
