# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for gim.irt — closed-form 2PL IRT scoring."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from gim.irt import DEFAULT_ITEM_BANK, GIMScorer, ScoringResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bank(items: dict[str, dict], sigma_hat: float = 1.0) -> dict:
    """Build an in-memory item bank in the schema GIMScorer expects."""
    return {
        "version": 1,
        "transform": {"squeeze_scale": 0.998, "squeeze_offset": 0.001},
        "items": items,
        "calibration": {"sigma_hat": sigma_hat},
    }


def _write_bank(tmp_path: Path, bank: dict) -> Path:
    p = tmp_path / "bank.json"
    p.write_text(json.dumps(bank))
    return p


def _scorer_from_items(tmp_path: Path, items: dict, sigma_hat: float = 1.0) -> GIMScorer:
    return GIMScorer(_write_bank(tmp_path, _bank(items, sigma_hat=sigma_hat)))


# ---------------------------------------------------------------------------
# Default item bank
# ---------------------------------------------------------------------------


class TestDefaultItemBank:
    def test_default_bank_path_exists(self):
        assert DEFAULT_ITEM_BANK.exists()

    def test_loads_default_bank(self):
        scorer = GIMScorer()
        assert scorer.n_items > 0

    def test_prompt_ids_returned_as_list(self):
        scorer = GIMScorer()
        ids = scorer.prompt_ids
        assert isinstance(ids, list)
        assert len(ids) == scorer.n_items
        # Returned list should be a copy, not the internal one
        ids.append("not_real")
        assert "not_real" not in scorer.prompt_ids


# ---------------------------------------------------------------------------
# Custom item bank loading
# ---------------------------------------------------------------------------


class TestCustomBank:
    def test_loads_from_path_string(self, tmp_path):
        p = _write_bank(tmp_path, _bank({"p1": {"a": 1.0, "b": 0.0}}))
        scorer = GIMScorer(str(p))
        assert scorer.n_items == 1
        assert scorer.prompt_ids == ["p1"]

    def test_loads_from_path_object(self, tmp_path):
        p = _write_bank(tmp_path, _bank({"p1": {"a": 1.0, "b": 0.0}}))
        scorer = GIMScorer(p)
        assert scorer.n_items == 1

    def test_prompt_ids_sorted(self, tmp_path):
        scorer = _scorer_from_items(
            tmp_path,
            {
                "z_last": {"a": 1.0, "b": 0.0},
                "a_first": {"a": 1.0, "b": 0.0},
                "m_mid": {"a": 1.0, "b": 0.0},
            },
        )
        assert scorer.prompt_ids == ["a_first", "m_mid", "z_last"]

    def test_default_sigma_hat_when_missing(self, tmp_path):
        bank = {
            "version": 1,
            "transform": {},
            "items": {"p1": {"a": 1.0, "b": 0.0}},
        }
        p = tmp_path / "no_calib.json"
        p.write_text(json.dumps(bank))
        scorer = GIMScorer(p)
        # sigma_hat falls back to 1.0
        assert scorer._sigma_hat == 1.0


# ---------------------------------------------------------------------------
# Logit transform
# ---------------------------------------------------------------------------


class TestLogitTransform:
    def test_half_maps_to_zero(self):
        result = GIMScorer._logit_transform(np.array([0.5]))
        assert abs(result[0]) < 1e-9

    def test_zero_is_finite(self):
        """Squeeze should keep p=0 inside (0, 1) so logit is finite."""
        result = GIMScorer._logit_transform(np.array([0.0]))
        assert np.isfinite(result[0])
        assert result[0] < 0  # logit(small) is negative

    def test_one_is_finite(self):
        """Squeeze should keep p=1 inside (0, 1) so logit is finite."""
        result = GIMScorer._logit_transform(np.array([1.0]))
        assert np.isfinite(result[0])
        assert result[0] > 0  # logit(near-1) is positive

    def test_monotonic(self):
        result = GIMScorer._logit_transform(np.array([0.1, 0.3, 0.5, 0.7, 0.9]))
        assert np.all(np.diff(result) > 0)


# ---------------------------------------------------------------------------
# score() — empty / no-overlap behavior
# ---------------------------------------------------------------------------


class TestScoreEmpty:
    def test_empty_dict_returns_nan(self, tmp_path):
        scorer = _scorer_from_items(tmp_path, {"p1": {"a": 1.0, "b": 0.0}})
        result = scorer.score({})
        assert math.isnan(result.theta)
        assert math.isnan(result.se)
        assert result.n_items_scored == 0
        assert result.coverage == 0.0
        assert result.n_items_total == 1

    def test_no_overlap_returns_nan(self, tmp_path):
        scorer = _scorer_from_items(tmp_path, {"p1": {"a": 1.0, "b": 0.0}})
        result = scorer.score({"unknown_id": 0.5})
        assert math.isnan(result.theta)
        assert result.n_items_scored == 0
        assert result.coverage == 0.0

    def test_only_nan_scores_returns_nan(self, tmp_path):
        scorer = _scorer_from_items(tmp_path, {"p1": {"a": 1.0, "b": 0.0}})
        result = scorer.score({"p1": float("nan")})
        assert math.isnan(result.theta)
        assert result.n_items_scored == 0


# ---------------------------------------------------------------------------
# score() — closed-form WLS
# ---------------------------------------------------------------------------


class TestScoreClosedForm:
    def test_perfect_score_yields_positive_theta(self, tmp_path):
        scorer = _scorer_from_items(
            tmp_path, {"p1": {"a": 1.5, "b": 0.0}, "p2": {"a": 1.0, "b": 0.5}}
        )
        result = scorer.score({"p1": 1.0, "p2": 1.0})
        assert result.theta > 0
        assert result.n_items_scored == 2
        assert result.coverage == 1.0

    def test_failing_score_yields_negative_theta(self, tmp_path):
        scorer = _scorer_from_items(
            tmp_path, {"p1": {"a": 1.5, "b": 0.0}, "p2": {"a": 1.0, "b": 0.5}}
        )
        result = scorer.score({"p1": 0.0, "p2": 0.0})
        assert result.theta < 0

    def test_se_is_positive(self, tmp_path):
        scorer = _scorer_from_items(
            tmp_path, {"p1": {"a": 1.5, "b": 0.0}}, sigma_hat=2.0
        )
        result = scorer.score({"p1": 0.7})
        assert result.se > 0

    def test_se_decreases_with_more_items(self, tmp_path):
        """SE = σ̂ / sqrt(Σ a²) — more items (same a) → smaller SE."""
        one_path = tmp_path / "one.json"
        one_path.write_text(json.dumps(_bank({"p1": {"a": 1.0, "b": 0.0}})))
        many_path = tmp_path / "many.json"
        many_path.write_text(
            json.dumps(_bank({f"p{i}": {"a": 1.0, "b": 0.0} for i in range(10)}))
        )
        bank_one = GIMScorer(one_path)
        bank_many = GIMScorer(many_path)
        r1 = bank_one.score({"p1": 0.7})
        r10 = bank_many.score({f"p{i}": 0.7 for i in range(10)})
        assert r10.se < r1.se

    def test_ci_brackets_theta(self, tmp_path):
        scorer = _scorer_from_items(tmp_path, {"p1": {"a": 1.0, "b": 0.0}})
        result = scorer.score({"p1": 0.5})
        assert result.ci_95_lower < result.theta < result.ci_95_upper
        assert math.isclose(
            result.ci_95_upper - result.theta,
            result.theta - result.ci_95_lower,
            rel_tol=1e-9,
        )

    def test_ci_width_matches_1_96_se(self, tmp_path):
        scorer = _scorer_from_items(tmp_path, {"p1": {"a": 1.0, "b": 0.0}}, sigma_hat=1.5)
        result = scorer.score({"p1": 0.5})
        assert math.isclose(
            result.ci_95_upper - result.theta, 1.96 * result.se, rel_tol=1e-9
        )

    def test_theta_at_difficulty_when_p_half(self, tmp_path):
        """When p=0.5 (logit 0) and b is fixed, θ ≈ b for a single item."""
        scorer = _scorer_from_items(tmp_path, {"p1": {"a": 1.0, "b": 1.2}})
        result = scorer.score({"p1": 0.5})
        # y_squeezed = logit(0.5*0.998 + 0.001) = logit(0.5) ≈ 0
        # θ = (a*(y + a*b)) / a² = (y + a*b)/a ≈ b
        assert math.isclose(result.theta, 1.2, abs_tol=1e-3)

    def test_higher_discrimination_dominates(self, tmp_path):
        """Item with higher a should pull θ toward its implied value."""
        scorer = _scorer_from_items(
            tmp_path,
            {
                "high_a": {"a": 5.0, "b": 1.0},
                "low_a": {"a": 0.5, "b": -1.0},
            },
        )
        result = scorer.score({"high_a": 0.5, "low_a": 0.5})
        # high_a item with p=0.5 implies θ ≈ b = 1.0
        # low_a item with p=0.5 implies θ ≈ b = -1.0
        # Closed-form WLS weights by a², so θ ≈ 1.0
        assert result.theta > 0.5

    def test_coverage_partial(self, tmp_path):
        scorer = _scorer_from_items(
            tmp_path,
            {f"p{i}": {"a": 1.0, "b": 0.0} for i in range(4)},
        )
        result = scorer.score({"p0": 0.5, "p1": 0.5})
        assert result.n_items_scored == 2
        assert result.coverage == 0.5

    def test_nan_scores_filtered(self, tmp_path):
        scorer = _scorer_from_items(
            tmp_path, {"p1": {"a": 1.0, "b": 0.0}, "p2": {"a": 1.0, "b": 0.0}}
        )
        result = scorer.score({"p1": 0.5, "p2": float("nan")})
        assert result.n_items_scored == 1

    def test_unknown_ids_silently_ignored(self, tmp_path):
        scorer = _scorer_from_items(tmp_path, {"p1": {"a": 1.0, "b": 0.0}})
        result = scorer.score({"p1": 0.5, "ghost_id": 0.99})
        assert result.n_items_scored == 1

    def test_raw_mean_matches_input(self, tmp_path):
        scorer = _scorer_from_items(
            tmp_path, {"p1": {"a": 1.0, "b": 0.0}, "p2": {"a": 1.0, "b": 0.0}}
        )
        result = scorer.score({"p1": 0.2, "p2": 0.8})
        assert math.isclose(result.raw_mean, 0.5, abs_tol=1e-9)

    def test_mse_fit_zero_for_single_item(self, tmp_path):
        """With one item, the closed-form θ fits exactly → mse=0."""
        scorer = _scorer_from_items(tmp_path, {"p1": {"a": 1.5, "b": 0.3}})
        result = scorer.score({"p1": 0.7})
        assert math.isclose(result.mse_fit, 0.0, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# ScoringResult dataclass behavior
# ---------------------------------------------------------------------------


class TestScoringResult:
    def test_to_dict_has_expected_keys(self):
        r = ScoringResult(
            theta=0.5,
            se=0.1,
            ci_95_lower=0.3,
            ci_95_upper=0.7,
            n_items_scored=10,
            n_items_total=20,
            coverage=0.5,
            raw_mean=0.6,
            mse_fit=0.01,
        )
        d = r.to_dict()
        assert d["theta"] == 0.5
        assert d["se_theta"] == 0.1  # renamed in to_dict
        assert d["ci_95_lower"] == 0.3
        assert d["ci_95_upper"] == 0.7
        assert d["n_items_scored"] == 10
        assert d["n_items_total"] == 20
        assert d["coverage"] == 0.5
        assert d["raw_mean"] == 0.6
        assert d["mse_fit"] == 0.01

    def test_repr_contains_summary_fields(self):
        r = ScoringResult(
            theta=0.5,
            se=0.1,
            ci_95_lower=0.3,
            ci_95_upper=0.7,
            n_items_scored=10,
            n_items_total=20,
            coverage=0.5,
            raw_mean=0.6,
            mse_fit=0.01,
        )
        s = repr(r)
        assert "ScoringResult" in s
        assert "10/20" in s
        assert "raw_mean" in s
