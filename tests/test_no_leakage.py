"""Тесты защиты от leakage (спецификация, разделы 9.4 и 17).

Проверяется контракт списков колонок и то, что пайплайн падает,
если запрещённое поле попало в матрицу признаков.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import pandas as pd
import pytest

from xg_context.config import (
    BASELINE_FEATURES,
    BENCHMARK_COLUMNS,
    CATEGORICAL_FEATURES,
    DEFENSIVE_CONTEXT_FEATURES,
    FEATURE_SETS,
    FORBIDDEN_FEATURE_COLUMNS,
    PROCESSED_DATA_DIR,
    SHOT_CONTEXT_FEATURES,
    TARGET_COLUMN,
)
from xg_context.features import ForbiddenColumnError, assert_no_forbidden_columns
from xg_context.preprocessing import build_preprocessor, select_features

ALL_FEATURES = (*BASELINE_FEATURES, *SHOT_CONTEXT_FEATURES, *DEFENSIVE_CONTEXT_FEATURES)


class TestFeatureListContract:
    def test_no_feature_is_forbidden(self) -> None:
        overlap = set(ALL_FEATURES) & set(FORBIDDEN_FEATURE_COLUMNS)
        assert overlap == set(), f"Признаки пересекаются с запрещёнными полями: {overlap}"

    def test_target_is_not_a_feature(self) -> None:
        assert TARGET_COLUMN not in ALL_FEATURES

    def test_target_is_forbidden_as_a_feature(self) -> None:
        assert TARGET_COLUMN in FORBIDDEN_FEATURE_COLUMNS

    def test_statsbomb_xg_is_benchmark_and_forbidden_as_feature(self) -> None:
        """statsbomb_xg — только внешний benchmark (раздел 9.4)."""
        assert "statsbomb_xg" in BENCHMARK_COLUMNS
        assert "statsbomb_xg" in FORBIDDEN_FEATURE_COLUMNS
        assert "statsbomb_xg" not in ALL_FEATURES

    def test_feature_lists_have_no_duplicates(self) -> None:
        assert len(ALL_FEATURES) == len(set(ALL_FEATURES))

    def test_feature_groups_do_not_overlap(self) -> None:
        assert set(BASELINE_FEATURES).isdisjoint(SHOT_CONTEXT_FEATURES)
        assert set(BASELINE_FEATURES).isdisjoint(DEFENSIVE_CONTEXT_FEATURES)
        assert set(SHOT_CONTEXT_FEATURES).isdisjoint(DEFENSIVE_CONTEXT_FEATURES)

    @pytest.mark.parametrize(
        "column",
        ["end_location", "shot_end_x", "player_id", "team_id", "shot_outcome", "is_goal"],
    )
    def test_key_leakage_fields_are_listed(self, column: str) -> None:
        """Исход, траектория и идентичность обязаны быть в запрещённых."""
        assert column in FORBIDDEN_FEATURE_COLUMNS


class TestForbiddenColumnGuard:
    def test_clean_feature_matrix_passes(self) -> None:
        frame = pd.DataFrame(columns=list(ALL_FEATURES))
        # Функция ничего не возвращает: успех — отсутствие исключения.
        assert_no_forbidden_columns(frame)

    @pytest.mark.parametrize(
        "column", ["statsbomb_xg", "is_goal", "end_location", "player_id", "team_name"]
    )
    def test_direct_forbidden_column_raises(self, column: str) -> None:
        frame = pd.DataFrame(columns=[*BASELINE_FEATURES, column])
        with pytest.raises(ForbiddenColumnError, match=column):
            assert_no_forbidden_columns(frame)

    def test_one_hot_derivative_is_caught(self) -> None:
        """Пайплайн должен ловить и производные вида `shot_outcome_Goal`."""
        frame = pd.DataFrame(columns=[*BASELINE_FEATURES, "shot_outcome_Goal"])
        with pytest.raises(ForbiddenColumnError):
            assert_no_forbidden_columns(frame)

    def test_column_transformer_prefix_is_caught(self) -> None:
        """`ColumnTransformer` добавляет префикс `<name>__`, он не должен маскировать поле."""
        frame = pd.DataFrame(columns=["num__shot_distance", "num__statsbomb_xg"])
        with pytest.raises(ForbiddenColumnError):
            assert_no_forbidden_columns(frame)

    def test_similar_but_legitimate_name_is_allowed(self) -> None:
        """Похожее по написанию, но легитимное имя не должно вызывать ложное срабатывание."""
        assert_no_forbidden_columns(pd.DataFrame(columns=["goalkeeper_distance_to_shot"]))

    def test_accepts_plain_sequence_of_names(self) -> None:
        with pytest.raises(ForbiddenColumnError):
            assert_no_forbidden_columns(["shot_distance", "statsbomb_xg"])

    def test_error_message_names_the_context(self) -> None:
        with pytest.raises(ForbiddenColumnError, match="матрица X_train"):
            assert_no_forbidden_columns(["is_goal"], context="матрица X_train")


# --------------------------------------------------------------------------------------
# Расширенные проверки этапов 2–4: наборы признаков, препроцессор и готовый датасет
# --------------------------------------------------------------------------------------


class TestFeatureSetsAreClean:
    """Ни один набор из FEATURE_SETS не должен содержать запрещённых полей."""

    @pytest.mark.parametrize("name", sorted(FEATURE_SETS))
    def test_feature_set_has_no_forbidden_columns(self, name: str) -> None:
        assert_no_forbidden_columns(list(FEATURE_SETS[name]), context=f"набор {name}")

    @pytest.mark.parametrize("name", sorted(FEATURE_SETS))
    def test_feature_set_has_no_duplicates(self, name: str) -> None:
        features = FEATURE_SETS[name]
        assert len(features) == len(set(features))

    @pytest.mark.parametrize("name", sorted(FEATURE_SETS))
    def test_benchmark_never_enters_a_feature_set(self, name: str) -> None:
        assert "statsbomb_xg" not in FEATURE_SETS[name]

    def test_league_identity_is_excluded_from_features(self) -> None:
        """Решение владельца: лига используется для разбиения, но не как признак."""
        for name, features in FEATURE_SETS.items():
            assert "competition_id" not in features, name
            assert "competition_name" not in features, name

    def test_ablation_levels_are_strictly_nested(self) -> None:
        """Каждый уровень ablation обязан ДОБАВЛЯТЬ признаки, а не заменять их.

        Иначе разница метрик смешала бы эффект добавления с эффектом удаления.
        """
        levels = [
            "geometry",
            "geometry_shot",
            "geometry_shot_flags",
            "geometry_shot_flags_defensive",
        ]
        for lower, upper in pairwise(levels):
            assert set(FEATURE_SETS[lower]) < set(FEATURE_SETS[upper]), (
                f"{lower} не является строгим подмножеством {upper}"
            )

    def test_defensive_level_adds_exactly_the_defensive_group(self) -> None:
        added = set(FEATURE_SETS["geometry_shot_flags_defensive"]) - set(
            FEATURE_SETS["geometry_shot_flags"]
        )
        assert added == set(DEFENSIVE_CONTEXT_FEATURES)


class TestPreprocessorRejectsLeakage:
    """Пайплайн обязан падать до обучения, а не портить результат тихо."""

    def test_build_preprocessor_rejects_forbidden_feature(self) -> None:
        with pytest.raises(ForbiddenColumnError):
            build_preprocessor([*BASELINE_FEATURES, "statsbomb_xg"])

    def test_build_preprocessor_accepts_clean_features(self) -> None:
        build_preprocessor(list(FEATURE_SETS["geometry_shot_flags_defensive"]))

    def test_select_features_rejects_forbidden_column(self) -> None:
        frame = pd.DataFrame({"shot_distance": [1.0], "shot_angle": [0.5], "is_goal": [1]})
        with pytest.raises(ForbiddenColumnError):
            select_features(frame, ["shot_distance", "is_goal"])

    def test_select_features_reports_missing_columns(self) -> None:
        frame = pd.DataFrame({"shot_distance": [1.0]})
        with pytest.raises(KeyError, match="shot_angle"):
            select_features(frame, ["shot_distance", "shot_angle"])

    def test_select_features_returns_only_requested_columns(self) -> None:
        frame = pd.DataFrame({"shot_distance": [1.0], "shot_angle": [0.5], "statsbomb_xg": [0.1]})
        matrix = select_features(frame, ["shot_distance", "shot_angle"])
        assert list(matrix.columns) == ["shot_distance", "shot_angle"]


class TestTransformedMatrixIsClean:
    """После ColumnTransformer имена меняются — проверяем и их."""

    def test_transformed_feature_names_contain_no_forbidden_fields(self) -> None:
        features = list(FEATURE_SETS["geometry_shot_flags_defensive"])
        frame = pd.DataFrame(
            {
                **{f: [1.0, 2.0] for f in features if f not in CATEGORICAL_FEATURES},
                **{f: ["Right Foot", "Head"] for f in features if f in CATEGORICAL_FEATURES},
            }
        )
        preprocessor = build_preprocessor(features)
        preprocessor.fit(frame)
        names = list(preprocessor.get_feature_names_out())
        assert_no_forbidden_columns(names, context="матрица после ColumnTransformer")


class TestBuiltDatasetIsClean:
    """Проверка на реальном датасете, если он уже построен."""

    @staticmethod
    def _dataset_path() -> Path:
        return PROCESSED_DATA_DIR / "context_eligible_shots.parquet"

    def test_dataset_columns_cover_all_features(self) -> None:
        path = self._dataset_path()
        if not path.exists():
            pytest.skip("Датасет ещё не построен: запустите scripts/build_dataset.py")
        columns = set(pd.read_parquet(path, columns=None).columns)
        for name, features in FEATURE_SETS.items():
            missing = set(features) - columns
            assert not missing, f"В датасете нет признаков набора {name}: {sorted(missing)}"

    def test_feature_matrix_from_real_data_is_clean(self) -> None:
        path = self._dataset_path()
        if not path.exists():
            pytest.skip("Датасет ещё не построен: запустите scripts/build_dataset.py")
        frame = pd.read_parquet(path)
        matrix = select_features(frame, FEATURE_SETS["geometry_shot_flags_defensive"])
        assert_no_forbidden_columns(matrix, context="матрица X из реального датасета")

    def test_target_is_binary_and_matches_outcome(self) -> None:
        path = self._dataset_path()
        if not path.exists():
            pytest.skip("Датасет ещё не построен: запустите scripts/build_dataset.py")
        frame = pd.read_parquet(path, columns=["is_goal", "shot_outcome"])
        assert set(frame["is_goal"].unique()) <= {0, 1}
        goals = frame.loc[frame["shot_outcome"] == "Goal", "is_goal"]
        assert (goals == 1).all()
        others = frame.loc[frame["shot_outcome"] != "Goal", "is_goal"]
        assert (others == 0).all()

    def test_no_penalties_survived_the_filters(self) -> None:
        path = self._dataset_path()
        if not path.exists():
            pytest.skip("Датасет ещё не построен: запустите scripts/build_dataset.py")
        frame = pd.read_parquet(path, columns=["shot_type", "period"])
        assert not (frame["shot_type"] == "Penalty").any()
        assert not (frame["period"] == 5).any()
