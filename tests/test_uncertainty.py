"""Тесты признака устойчивости разницы метрик.

Разница устойчива, если доверительный интервал целиком лежит по одну сторону от нуля.
Знак при этом не важен: устойчиво хуже это такой же результат, как устойчиво лучше.
Именно на этом раньше терялся вывод про Brier score.
Там наша модель устойчиво уступает statsbomb_xg.
"""

from __future__ import annotations

import numpy as np
import pytest

from xg_context.evaluation import interval_excludes_zero, paired_bootstrap_by_match


class TestIntervalExcludesZero:
    def test_negative_interval_is_stable(self) -> None:
        assert interval_excludes_zero(-0.0160, -0.0078)

    def test_positive_interval_is_stable(self) -> None:
        assert interval_excludes_zero(0.0003, 0.0022)

    def test_interval_crossing_zero_is_not_stable(self) -> None:
        assert not interval_excludes_zero(-0.0009, 0.0065)

    def test_interval_touching_zero_is_not_stable(self) -> None:
        assert not interval_excludes_zero(0.0, 0.0065)
        assert not interval_excludes_zero(-0.0065, 0.0)

    def test_nan_interval_is_not_stable(self) -> None:
        assert not interval_excludes_zero(float("nan"), 0.01)
        assert not interval_excludes_zero(-0.01, float("nan"))


class TestBootstrapReportsBothDirections:
    """Bootstrap обязан помечать устойчивым и ухудшение, а не только улучшение."""

    @pytest.fixture
    def data(self) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(0)
        n_matches, per_match = 60, 25
        matches = np.repeat(np.arange(n_matches), per_match)
        truth = rng.random(len(matches)) < 0.12
        return {
            "y": truth.astype(int),
            "matches": matches,
            "good": np.where(truth, 0.30, 0.06),
            "bad": np.full(len(matches), 0.12),
        }

    def test_worse_candidate_is_marked_stable(self, data) -> None:
        stats = paired_bootstrap_by_match(
            data["y"], data["good"], data["bad"], data["matches"], n_bootstrap=200, seed=1
        )
        assert stats["delta_log_loss"] > 0
        assert stats["delta_log_loss_ci_low"] > 0
        assert stats["delta_log_loss_significant"]
        assert stats["delta_brier_significant"]

    def test_better_candidate_is_marked_stable(self, data) -> None:
        stats = paired_bootstrap_by_match(
            data["y"], data["bad"], data["good"], data["matches"], n_bootstrap=200, seed=1
        )
        assert stats["delta_log_loss"] < 0
        assert stats["delta_log_loss_ci_high"] < 0
        assert stats["delta_log_loss_significant"]

    def test_identical_models_are_not_stable(self, data) -> None:
        stats = paired_bootstrap_by_match(
            data["y"], data["bad"], data["bad"], data["matches"], n_bootstrap=200, seed=1
        )
        assert stats["delta_log_loss"] == pytest.approx(0.0, abs=1e-12)
        assert not stats["delta_log_loss_significant"]
        assert not stats["delta_brier_significant"]
