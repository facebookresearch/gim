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
from gim.judges import PERMITTED_JUDGES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


TEST_JUDGE = "test-judge"


def _bank(
    items: dict[str, dict],
    sigma_hat: float = 1.0,
    judge_effects: dict[str, float] | None = None,
) -> dict:
    """Build an in-memory item bank in the schema GIMScorer expects.

    Defaults to a single judge with gamma=0 so closed-form assertions are
    unaffected by the judge adjustment.
    """
    judge_effects = judge_effects if judge_effects is not None else {TEST_JUDGE: 0.0}
    return {
        "version": 2,
        "model": "2PL_logit_normal_judge_fixed_effect",
        "transform": {"squeeze_scale": 0.998, "squeeze_offset": 0.001},
        "scoring": {"requires_judge_id": True},
        "items": items,
        "calibration": {"sigma_hat": sigma_hat},
        "judge_effects": judge_effects,
        "permitted_judges": {
            judge_id: {"display_name": judge_id, "gamma": gamma}
            for judge_id, gamma in judge_effects.items()
        },
    }


def _write_bank(tmp_path: Path, bank: dict) -> Path:
    p = tmp_path / "bank.json"
    p.write_text(json.dumps(bank))
    return p


def _scorer_from_items(
    tmp_path: Path, items: dict, sigma_hat: float = 1.0
) -> GIMScorer:
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


class TestDefaultBankJudgeRegistry:
    """The shipped bank and the judge registry must not drift apart.

    resolve_judge() gates on PERMITTED_JUDGES before inference, while gamma
    comes from the bank at metric time. If the two disagree, a run is admitted
    and then fails after the full inference spend.
    """

    def test_judge_ids_match_registry(self):
        assert set(GIMScorer().permitted_judges) == set(PERMITTED_JUDGES)

    def test_judge_routes_and_names_match_registry(self):
        bank_judges = GIMScorer().permitted_judges
        for judge_id, spec in PERMITTED_JUDGES.items():
            assert bank_judges[judge_id]["default_model"] == spec.default_model
            assert bank_judges[judge_id]["display_name"] == spec.display_name

    def test_every_registry_judge_has_a_finite_gamma(self):
        judge_effects = GIMScorer().judge_effects
        for judge_id in PERMITTED_JUDGES:
            assert math.isfinite(judge_effects[judge_id])

    def test_judge_effects_are_centered(self):
        # The bank declares mean(gamma_k) = 0; an uncentered bank would shift
        # every theta off the published scale.
        gammas = list(GIMScorer().judge_effects.values())
        assert math.isclose(sum(gammas) / len(gammas), 0.0, abs_tol=1e-6)


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
        bank = _bank({"p1": {"a": 1.0, "b": 0.0}})
        del bank["calibration"]
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
        result = scorer.score({}, judge=TEST_JUDGE)
        assert math.isnan(result.theta)
        assert math.isnan(result.se)
        assert result.n_items_scored == 0
        assert result.coverage == 0.0
        assert result.n_items_total == 1

    def test_no_overlap_returns_nan(self, tmp_path):
        scorer = _scorer_from_items(tmp_path, {"p1": {"a": 1.0, "b": 0.0}})
        result = scorer.score({"unknown_id": 0.5}, judge=TEST_JUDGE)
        assert math.isnan(result.theta)
        assert result.n_items_scored == 0
        assert result.coverage == 0.0

    def test_only_nan_scores_returns_nan(self, tmp_path):
        scorer = _scorer_from_items(tmp_path, {"p1": {"a": 1.0, "b": 0.0}})
        result = scorer.score({"p1": float("nan")}, judge=TEST_JUDGE)
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
        result = scorer.score({"p1": 1.0, "p2": 1.0}, judge=TEST_JUDGE)
        assert result.theta > 0
        assert result.n_items_scored == 2
        assert result.coverage == 1.0

    def test_failing_score_yields_negative_theta(self, tmp_path):
        scorer = _scorer_from_items(
            tmp_path, {"p1": {"a": 1.5, "b": 0.0}, "p2": {"a": 1.0, "b": 0.5}}
        )
        result = scorer.score({"p1": 0.0, "p2": 0.0}, judge=TEST_JUDGE)
        assert result.theta < 0

    def test_se_is_positive(self, tmp_path):
        scorer = _scorer_from_items(
            tmp_path, {"p1": {"a": 1.5, "b": 0.0}}, sigma_hat=2.0
        )
        result = scorer.score({"p1": 0.7}, judge=TEST_JUDGE)
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
        r1 = bank_one.score({"p1": 0.7}, judge=TEST_JUDGE)
        r10 = bank_many.score({f"p{i}": 0.7 for i in range(10)}, judge=TEST_JUDGE)
        assert r10.se < r1.se

    def test_ci_brackets_theta(self, tmp_path):
        scorer = _scorer_from_items(tmp_path, {"p1": {"a": 1.0, "b": 0.0}})
        result = scorer.score({"p1": 0.5}, judge=TEST_JUDGE)
        assert result.ci_95_lower < result.theta < result.ci_95_upper
        assert math.isclose(
            result.ci_95_upper - result.theta,
            result.theta - result.ci_95_lower,
            rel_tol=1e-9,
        )

    def test_ci_width_matches_1_96_se(self, tmp_path):
        scorer = _scorer_from_items(
            tmp_path, {"p1": {"a": 1.0, "b": 0.0}}, sigma_hat=1.5
        )
        result = scorer.score({"p1": 0.5}, judge=TEST_JUDGE)
        assert math.isclose(
            result.ci_95_upper - result.theta, 1.96 * result.se, rel_tol=1e-9
        )

    def test_theta_at_difficulty_when_p_half(self, tmp_path):
        """When p=0.5 (logit 0) and b is fixed, θ ≈ b for a single item."""
        scorer = _scorer_from_items(tmp_path, {"p1": {"a": 1.0, "b": 1.2}})
        result = scorer.score({"p1": 0.5}, judge=TEST_JUDGE)
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
        result = scorer.score({"high_a": 0.5, "low_a": 0.5}, judge=TEST_JUDGE)
        # high_a item with p=0.5 implies θ ≈ b = 1.0
        # low_a item with p=0.5 implies θ ≈ b = -1.0
        # Closed-form WLS weights by a², so θ ≈ 1.0
        assert result.theta > 0.5

    def test_coverage_partial(self, tmp_path):
        scorer = _scorer_from_items(
            tmp_path,
            {f"p{i}": {"a": 1.0, "b": 0.0} for i in range(4)},
        )
        result = scorer.score({"p0": 0.5, "p1": 0.5}, judge=TEST_JUDGE)
        assert result.n_items_scored == 2
        assert result.coverage == 0.5

    def test_nan_scores_filtered(self, tmp_path):
        scorer = _scorer_from_items(
            tmp_path, {"p1": {"a": 1.0, "b": 0.0}, "p2": {"a": 1.0, "b": 0.0}}
        )
        result = scorer.score({"p1": 0.5, "p2": float("nan")}, judge=TEST_JUDGE)
        assert result.n_items_scored == 1

    def test_unknown_ids_silently_ignored(self, tmp_path):
        scorer = _scorer_from_items(tmp_path, {"p1": {"a": 1.0, "b": 0.0}})
        result = scorer.score({"p1": 0.5, "ghost_id": 0.99}, judge=TEST_JUDGE)
        assert result.n_items_scored == 1

    def test_raw_mean_matches_input(self, tmp_path):
        scorer = _scorer_from_items(
            tmp_path, {"p1": {"a": 1.0, "b": 0.0}, "p2": {"a": 1.0, "b": 0.0}}
        )
        result = scorer.score({"p1": 0.2, "p2": 0.8}, judge=TEST_JUDGE)
        assert math.isclose(result.raw_mean, 0.5, abs_tol=1e-9)

    def test_mse_fit_zero_for_single_item(self, tmp_path):
        """With one item, the closed-form θ fits exactly → mse=0."""
        scorer = _scorer_from_items(tmp_path, {"p1": {"a": 1.5, "b": 0.3}})
        result = scorer.score({"p1": 0.7}, judge=TEST_JUDGE)
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


# ---------------------------------------------------------------------------
# Judge fixed effects
# ---------------------------------------------------------------------------


_JUDGE_ITEMS = {"p1": {"a": 1.0, "b": 0.5}, "p2": {"a": 2.0, "b": -0.25}}
_TWO_JUDGES = {"strict": -0.4, "lenient": 0.4}


def _raw_from_logit(y: float) -> float:
    """Invert the squeeze+logit transform so a target y can be requested."""
    squeezed = np.exp(y) / (1.0 + np.exp(y))
    return float((squeezed - 0.001) / 0.998)


class TestJudgeEffects:
    def test_score_observations_subtracts_judge_effects(self, tmp_path):
        scorer = GIMScorer(
            _write_bank(
                tmp_path,
                _bank(_JUDGE_ITEMS, sigma_hat=2.0, judge_effects=_TWO_JUDGES),
            )
        )

        theta = 1.25
        y1 = 1.0 * (theta - 0.5) - 0.4
        y2 = 2.0 * (theta - (-0.25)) + 0.4
        result = scorer.score_observations(
            [
                ("p1", _raw_from_logit(y1), "strict"),
                ("p2", _raw_from_logit(y2), "lenient"),
            ]
        )

        assert result.theta == pytest.approx(theta)
        assert result.n_items_scored == 2
        assert result.n_observations == 2
        assert result.n_judges == 2

    def test_item_bank_requires_judge_id_scoring_flag(self, tmp_path):
        bank = _bank(_JUDGE_ITEMS)
        del bank["scoring"]["requires_judge_id"]
        with pytest.raises(ValueError, match="requires_judge_id"):
            GIMScorer(_write_bank(tmp_path, bank))

    def test_item_bank_must_use_judge_fixed_effect_model(self, tmp_path):
        bank = _bank(_JUDGE_ITEMS)
        bank["model"] = "2PL_logit_normal"
        with pytest.raises(ValueError, match="judge fixed-effect"):
            GIMScorer(_write_bank(tmp_path, bank))

    def test_item_bank_requires_judge_metadata(self, tmp_path):
        bank = _bank(_JUDGE_ITEMS)
        del bank["permitted_judges"]
        with pytest.raises(ValueError, match="permitted_judges"):
            GIMScorer(_write_bank(tmp_path, bank))

    def test_requires_known_judge_for_scoring(self, tmp_path):
        scorer = GIMScorer(
            _write_bank(tmp_path, _bank(_JUDGE_ITEMS, judge_effects=_TWO_JUDGES))
        )

        with pytest.raises(ValueError, match="requires a judge_id"):
            scorer.score({"p1": 0.5})

        with pytest.raises(ValueError, match="Unknown judge_id"):
            scorer.score({"p1": 0.5}, judge="unknown")

    def test_score_observations_rejects_unlabeled_tuples(self, tmp_path):
        scorer = GIMScorer(_write_bank(tmp_path, _bank(_JUDGE_ITEMS)))

        with pytest.raises(ValueError, match="prompt_id, score, judge"):
            scorer.score_observations([("p1", 0.5)])

    def test_permitted_judges_reflect_item_bank(self, tmp_path):
        scorer = GIMScorer(
            _write_bank(tmp_path, _bank(_JUDGE_ITEMS, judge_effects=_TWO_JUDGES))
        )

        assert scorer.judge_effects == {"strict": -0.4, "lenient": 0.4}
        assert scorer.permitted_judges == {
            "strict": {"display_name": "strict", "gamma": -0.4},
            "lenient": {"display_name": "lenient", "gamma": 0.4},
        }
