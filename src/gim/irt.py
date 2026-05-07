# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""GIM IRT scoring module.

Computes model ability (θ) from GIM scores using a pre-calibrated 2PL IRT
item bank.  Uses only numpy — no PyTorch, no GPU, no optimization loop.

The 2PL IRT model is:  y_j = a_j * (θ - b_j)

where y_j is the logit-transformed score. Given fixed (a_j, b_j), the optimal θ
is a closed-form weighted least squares solution:

    θ* = Σ a_j (y_j + a_j b_j) / Σ a_j²
    SE(θ) = σ̂ / √(Σ a_j²)

where σ̂ is the calibration residual standard deviation stored in the item bank.

Usage::

    from gim.irt import GIMScorer

    scorer = GIMScorer()  # loads bundled 615-item public bank
    result = scorer.score({"p00079911": 0.85, "p00217491": 0.42, ...})
    print(result)  # ScoringResult(theta=0.73, se=0.024, n_items=615, ...)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np

# Default item bank bundled with the package.
DEFAULT_ITEM_BANK = Path(__file__).parent / "irt_params.json"


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

    def __repr__(self) -> str:
        return (
            f"ScoringResult(θ={self.theta:+.4f} ± {self.se:.4f}, "
            f"95% CI=[{self.ci_95_lower:+.4f}, {self.ci_95_upper:+.4f}], "
            f"items={self.n_items_scored}/{self.n_items_total}, "
            f"raw_mean={self.raw_mean:.4f})"
        )

    def to_dict(self) -> dict:
        return {
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


class GIMScorer:
    """Score models against the GIM IRT item bank.

    Loads a pre-calibrated item bank (JSON) and computes ability (θ) for any
    set of prompt scores using a closed-form solution. No optimization needed.

    Args:
        item_bank_path: Path to an ``irt_item_bank*.json`` file. Defaults to
            the bundled 615-item public item bank.
    """

    def __init__(self, item_bank_path: Union[str, Path, None] = None):
        path = Path(item_bank_path) if item_bank_path is not None else DEFAULT_ITEM_BANK
        with open(path) as f:
            data = json.load(f)

        self._version = data["version"]
        self._transform = data["transform"]
        self._items = data["items"]

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

    @staticmethod
    def _logit_transform(p: np.ndarray) -> np.ndarray:
        """Squeeze + logit: maps [0, 1] scores to (-∞, +∞)."""
        p_squeezed = p * 0.998 + 0.001
        return np.log(p_squeezed / (1.0 - p_squeezed))

    def score(self, scores: dict[str, float]) -> ScoringResult:
        """Score a model from a dict of {prompt_id: raw_score}.

        Args:
            scores: Mapping from gim_prompt_id to raw score in [0, 1].
                    Missing prompts are simply excluded (not penalized).
                    NaN values are excluded.

        Returns:
            ScoringResult with theta, SE, confidence interval, and fit stats.
        """
        # Match to item bank
        matched_idx = []
        matched_scores = []
        for pid, raw_score in scores.items():
            if pid in self._pid_to_idx and not np.isnan(raw_score):
                matched_idx.append(self._pid_to_idx[pid])
                matched_scores.append(raw_score)

        if len(matched_idx) == 0:
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
            )

        idx = np.array(matched_idx)
        p = np.array(matched_scores)
        a = self._a[idx]
        b = self._b[idx]
        y = self._logit_transform(p)

        # Closed-form WLS: θ* = Σ a_j(y_j + a_j b_j) / Σ a_j²
        a_sq = a**2
        theta = float(np.sum(a * (y + a * b)) / np.sum(a_sq))
        se = float(self._sigma_hat / np.sqrt(np.sum(a_sq)))

        # Fit residuals
        y_pred = a * (theta - b)
        mse = float(np.mean((y - y_pred) ** 2))

        return ScoringResult(
            theta=theta,
            se=se,
            ci_95_lower=theta - 1.96 * se,
            ci_95_upper=theta + 1.96 * se,
            n_items_scored=len(idx),
            n_items_total=self.n_items,
            coverage=len(idx) / self.n_items,
            raw_mean=float(np.mean(p)),
            mse_fit=mse,
        )
