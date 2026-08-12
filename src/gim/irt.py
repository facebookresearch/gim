# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""GIM IRT scoring module.

Computes model ability (θ) from GIM scores using a pre-calibrated 2PL IRT
item bank.  Uses only numpy — no PyTorch, no GPU, no optimization loop.

GIM item banks use the judge fixed-effect model:

    y_jk = a_j * (θ - b_j) + γ_k

where y_jk is the logit-transformed score and γ_k is a centered judge fixed
effect. Given fixed (a_j, b_j, γ_k), the optimal θ is a closed-form weighted
least squares solution:

    θ* = Σ a_j (y_jk - γ_k + a_j b_j) / Σ a_j²
    SE(θ) = σ̂ / √(Σ a_j²)

where σ̂ is the calibration residual standard deviation stored in the item bank.

Every scoring observation must carry a known calibrated judge label, so that
the judge's leniency is removed rather than absorbed into θ.

Usage::

    from gim.irt import GIMScorer

    scorer = GIMScorer()  # loads bundled 615-item public bank
    result = scorer.score(
        {"p00079911": 0.85, "p00217491": 0.42, ...},
        judge="gemini-3-flash-preview",
    )
    print(result)  # ScoringResult(θ=0.73, se=0.024, items=615, ...)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Union

import numpy as np

# Default item bank bundled with the package.
DEFAULT_ITEM_BANK = Path(__file__).parent / "irt_params.json"


@dataclass
class ScoringObservation:
    """One judge-labeled score observation for a prompt."""

    prompt_id: str
    score: float
    judge: str | None = None


@dataclass
class ScoringResult:
    """Result of scoring a model against the GIM item bank."""

    theta: float
    se: float
    ci_95_lower: float
    ci_95_upper: float
    n_items_scored: int
    n_items_total: int
    coverage: float
    raw_mean: float
    mse_fit: float
    n_observations: int | None = None
    n_judges: int | None = None

    def __repr__(self) -> str:
        obs_part = (
            f", observations={self.n_observations}"
            if self.n_observations is not None
            and self.n_observations != self.n_items_scored
            else ""
        )
        return (
            f"ScoringResult(θ={self.theta:+.4f} ± {self.se:.4f}, "
            f"95% CI=[{self.ci_95_lower:+.4f}, {self.ci_95_upper:+.4f}], "
            f"items={self.n_items_scored}/{self.n_items_total}{obs_part}, "
            f"raw_mean={self.raw_mean:.4f})"
        )

    def to_dict(self) -> dict:
        out = {
            "theta": self.theta,
            "se_theta": self.se,
            "ci_95_lower": self.ci_95_lower,
            "ci_95_upper": self.ci_95_upper,
            "n_items_scored": self.n_items_scored,
            "n_items_total": self.n_items_total,
            "coverage": self.coverage,
            "raw_mean": self.raw_mean,
            "mse_fit": self.mse_fit,
        }
        if self.n_observations is not None:
            out["n_observations"] = self.n_observations
        if self.n_judges is not None:
            out["n_judges"] = self.n_judges
        return out


class GIMScorer:
    """Score models against the GIM IRT item bank.

    Loads a pre-calibrated judge-aware item bank and computes ability (θ) for
    prompt scores using a closed-form judge-adjusted solution.

    Args:
        item_bank_path: Path to an ``irt_item_bank*.json`` file. Defaults to
            the bundled 615-item public item bank.
    """

    def __init__(self, item_bank_path: Union[str, Path, None] = None):
        path = Path(item_bank_path) if item_bank_path is not None else DEFAULT_ITEM_BANK
        with open(path) as f:
            data = json.load(f)

        if data.get("model") != "2PL_logit_normal_judge_fixed_effect":
            raise ValueError("GIM scoring requires the judge fixed-effect IRT model.")
        scoring = data.get("scoring", {})
        if scoring.get("requires_judge_id") is not True:
            raise ValueError("GIM scoring requires scoring.requires_judge_id=true.")
        if not data.get("judge_effects"):
            raise ValueError("GIM scoring requires item-bank judge_effects.")
        if not data.get("permitted_judges"):
            raise ValueError("GIM scoring requires item-bank permitted_judges.")

        self._transform = data["transform"]
        self._items = data["items"]
        for prompt_id, item in self._items.items():
            if "a" not in item or "b" not in item:
                raise ValueError(
                    f"Item {prompt_id!r} is missing required IRT parameters a and b."
                )
        self._judge_effects = {
            str(k): float(v) for k, v in data["judge_effects"].items()
        }
        self._permitted_judges = data["permitted_judges"]

        # Pre-compute arrays keyed by prompt_id for fast lookup
        self._prompt_ids = sorted(self._items.keys())
        self._a = np.array([self._items[p]["a"] for p in self._prompt_ids])
        self._b = np.array([self._items[p]["b"] for p in self._prompt_ids])
        self._sigma_hat = data.get("calibration", {}).get("sigma_hat", 1.0)
        self._pid_to_idx = {p: i for i, p in enumerate(self._prompt_ids)}

    @property
    def n_items(self) -> int:
        return len(self._prompt_ids)

    @property
    def prompt_ids(self) -> list[str]:
        return list(self._prompt_ids)

    @property
    def judge_effects(self) -> dict[str, float]:
        return dict(self._judge_effects)

    @property
    def permitted_judges(self) -> dict:
        if self._permitted_judges:
            return dict(self._permitted_judges)
        return dict(self._judge_effects)

    @staticmethod
    def _logit_transform(p: np.ndarray) -> np.ndarray:
        """Squeeze + logit: maps [0, 1] scores to (-∞, +∞)."""
        p_squeezed = p * 0.998 + 0.001
        return np.log(p_squeezed / (1.0 - p_squeezed))

    def _empty_result(self) -> ScoringResult:
        return ScoringResult(
            theta=np.nan,
            se=np.nan,
            ci_95_lower=np.nan,
            ci_95_upper=np.nan,
            n_items_scored=0,
            n_items_total=self.n_items,
            coverage=0.0,
            raw_mean=np.nan,
            mse_fit=np.nan,
            n_observations=0,
            n_judges=0,
        )

    def _judge_gamma(self, judge: str | None) -> float:
        if judge is None:
            raise ValueError("This item bank requires a judge_id for IRT scoring.")
        judge_id = str(judge)
        if judge_id not in self._judge_effects:
            choices = ", ".join(sorted(self._judge_effects))
            raise ValueError(
                f"Unknown judge_id={judge_id!r} for this item bank. "
                f"Choose one of: {choices}."
            )
        return self._judge_effects[judge_id]

    def score(
        self,
        scores: dict[str, float],
        judge: str | None = None,
    ) -> ScoringResult:
        """Score a model from a dict of {prompt_id: raw_score}.

        Args:
            scores: Mapping from gim_prompt_id to raw score in [0, 1].
                    Missing prompts are simply excluded (not penalized).
                    NaN values are excluded.
            judge: Required calibrated judge label for all scores.

        Returns:
            ScoringResult with theta, SE, confidence interval, and fit stats.
        """
        observations = [
            ScoringObservation(prompt_id=pid, score=raw_score, judge=judge)
            for pid, raw_score in scores.items()
        ]
        return self.score_observations(observations)

    def score_observations(
        self,
        observations: Iterable[ScoringObservation | tuple],
    ) -> ScoringResult:
        """Score repeated judge-labeled observations.

        Each observation may be a ScoringObservation or a 3-tuple
        (prompt_id, score, judge). Multiple observations for the same prompt,
        including from different judges, are retained as independent
        measurements after judge adjustment.
        """
        matched_idx = []
        matched_scores = []
        matched_judges: list[str | None] = []

        for obs in observations:
            if isinstance(obs, ScoringObservation):
                pid = obs.prompt_id
                raw_score = obs.score
                judge = obs.judge
            else:
                if len(obs) == 3:
                    pid, raw_score, judge = obs
                else:
                    raise ValueError(
                        "observations must be ScoringObservation or "
                        "(prompt_id, score, judge)"
                    )

            try:
                raw_score_float = float(raw_score)
            except (TypeError, ValueError):
                continue
            if pid in self._pid_to_idx and not np.isnan(raw_score_float):
                matched_idx.append(self._pid_to_idx[str(pid)])
                matched_scores.append(raw_score_float)
                matched_judges.append(None if judge is None else str(judge))

        if len(matched_idx) == 0:
            return self._empty_result()

        idx = np.array(matched_idx)
        p = np.array(matched_scores)
        a = self._a[idx]
        b = self._b[idx]
        y = self._logit_transform(p)
        gamma = np.array([self._judge_gamma(judge) for judge in matched_judges])
        y_adjusted = y - gamma

        # Closed-form WLS after judge adjustment.
        a_sq = a**2
        theta = float(np.sum(a * (y_adjusted + a * b)) / np.sum(a_sq))
        se = float(self._sigma_hat / np.sqrt(np.sum(a_sq)))

        # Fit residuals
        y_pred = a * (theta - b) + gamma
        mse = float(np.mean((y - y_pred) ** 2))

        return ScoringResult(
            theta=theta,
            se=se,
            ci_95_lower=theta - 1.96 * se,
            ci_95_upper=theta + 1.96 * se,
            n_items_scored=len(set(idx.tolist())),
            n_items_total=self.n_items,
            coverage=len(set(idx.tolist())) / self.n_items,
            raw_mean=float(np.mean(p)),
            mse_fit=mse,
            n_observations=len(idx),
            n_judges=len({j for j in matched_judges if j is not None}),
        )
