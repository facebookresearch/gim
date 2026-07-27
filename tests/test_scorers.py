# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for gim.scorers — Pydantic models, scoring logic, and routing."""

import math
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from inspect_ai.model import ChatMessageAssistant, ChatMessageUser
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Target, mean_score
from inspect_ai.solver import TaskState

from gim.scorers import (
    ExactAnswerJudgment,
    RubricJudgment,
    EXACT_ANSWER_PROMPT,
    RUBRIC_GRADER_PROMPT,
    _score_exact_answer,
    _score_rubrics,
    gim_scorer,
)


# ---------------------------------------------------------------------------
# Pydantic schema tests
# ---------------------------------------------------------------------------


class TestExactAnswerJudgment:
    def test_valid_correct(self):
        j = ExactAnswerJudgment(explanation="Matches exactly", grade="CORRECT", confidence=0.9)
        assert j.grade == "CORRECT"
        assert j.confidence == 0.9

    def test_valid_incorrect(self):
        j = ExactAnswerJudgment(explanation="Does not match", grade="INCORRECT", confidence=0.8)
        assert j.grade == "INCORRECT"

    def test_unknown_grade_rejected(self):
        with pytest.raises(ValidationError):
            ExactAnswerJudgment(explanation="Ambiguous", grade="UNKNOWN", confidence=0.5)

    def test_invalid_grade(self):
        with pytest.raises(ValidationError):
            ExactAnswerJudgment(explanation="test", grade="MAYBE", confidence=0.5)

    def test_missing_confidence(self):
        with pytest.raises(ValidationError):
            ExactAnswerJudgment(explanation="test", grade="CORRECT")  # type: ignore[call-arg]

    def test_missing_explanation(self):
        with pytest.raises(ValidationError):
            ExactAnswerJudgment(grade="CORRECT", confidence=0.5)  # type: ignore[call-arg]

    def test_missing_grade(self):
        with pytest.raises(ValidationError):
            ExactAnswerJudgment(explanation="test", confidence=0.5)  # type: ignore[call-arg]

    def test_confidence_bounds(self):
        ExactAnswerJudgment(explanation="Min", grade="CORRECT", confidence=0.0)
        ExactAnswerJudgment(explanation="Max", grade="CORRECT", confidence=1.0)

    def test_confidence_below_zero(self):
        with pytest.raises(ValidationError):
            ExactAnswerJudgment(explanation="Bad", grade="CORRECT", confidence=-0.1)

    def test_confidence_above_one(self):
        with pytest.raises(ValidationError):
            ExactAnswerJudgment(explanation="Bad", grade="CORRECT", confidence=1.1)


class TestRubricJudgment:
    def test_valid(self):
        j = RubricJudgment(explanation="Good", score=0.8, confidence=0.9)
        assert j.score == 0.8
        assert j.confidence == 0.9

    def test_score_bounds(self):
        RubricJudgment(explanation="Min", score=0.0, confidence=0.5)
        RubricJudgment(explanation="Max", score=1.0, confidence=0.5)

    def test_score_below_zero(self):
        with pytest.raises(ValidationError):
            RubricJudgment(explanation="Bad", score=-0.1, confidence=0.5)

    def test_score_above_one(self):
        with pytest.raises(ValidationError):
            RubricJudgment(explanation="Bad", score=1.1, confidence=0.5)

    def test_confidence_below_zero(self):
        with pytest.raises(ValidationError):
            RubricJudgment(explanation="Bad", score=0.5, confidence=-0.1)

    def test_confidence_above_one(self):
        with pytest.raises(ValidationError):
            RubricJudgment(explanation="Bad", score=0.5, confidence=1.1)


# ---------------------------------------------------------------------------
# Prompt template tests
# ---------------------------------------------------------------------------


class TestPromptTemplates:
    def test_exact_answer_prompt_has_placeholders(self):
        assert "{question}" in EXACT_ANSWER_PROMPT
        assert "{answer}" in EXACT_ANSWER_PROMPT
        assert "{predicted_answer}" in EXACT_ANSWER_PROMPT

    def test_exact_answer_prompt_no_unknown_option(self):
        assert "UNKNOWN" not in EXACT_ANSWER_PROMPT

    def test_exact_answer_prompt_has_confidence_section(self):
        assert "# Confidence" in EXACT_ANSWER_PROMPT

    def test_exact_answer_prompt_formats(self):
        result = EXACT_ANSWER_PROMPT.format(
            question="Q?",
            answer="A",
            predicted_answer="A",
        )
        assert "Q?" in result
        assert "A" in result

    def test_rubric_prompt_has_placeholders(self):
        assert "{answer}" in RUBRIC_GRADER_PROMPT
        assert "{model_response}" in RUBRIC_GRADER_PROMPT
        assert "{rubric_string}" in RUBRIC_GRADER_PROMPT

    def test_rubric_prompt_formats(self):
        result = RUBRIC_GRADER_PROMPT.format(
            answer="gold",
            model_response="response",
            rubric_string="rubric",
        )
        assert "gold" in result
        assert "response" in result
        assert "rubric" in result


# ---------------------------------------------------------------------------
# Helpers for mocking Inspect AI internals
# ---------------------------------------------------------------------------


def _make_task_state(
    input_text: str,
    completion: str,
    metadata: dict | None = None,
) -> TaskState:
    """Create a minimal TaskState for scorer tests."""
    state = MagicMock(spec=TaskState)
    state.input_text = input_text
    state.output = MagicMock()
    state.output.completion = completion
    state.metadata = metadata or {}
    return state


def _mock_judge(response_json: str):
    """Patch _call_exact_judge / _call_rubric_judge to return parsed objects directly."""
    pass


def _mock_exact_judge(response_json: str):
    """Patch _call_exact_judge to return a parsed ExactAnswerJudgment."""
    judgment = ExactAnswerJudgment.model_validate_json(response_json)
    return patch("gim.scorers._call_exact_judge", new=AsyncMock(return_value=judgment))


def _mock_rubric_judge(response_json: str):
    """Patch _call_rubric_judge to return a parsed RubricJudgment."""
    judgment = RubricJudgment.model_validate_json(response_json)
    return patch("gim.scorers._call_rubric_judge", new=AsyncMock(return_value=judgment))


def _mock_rubric_judge_side_effect(side_effect):
    """Patch _call_rubric_judge with a side_effect list."""
    return patch("gim.scorers._call_rubric_judge", new=AsyncMock(side_effect=side_effect))


def _mock_exact_judge_raises(exc):
    """Patch _call_exact_judge to raise an exception (simulating all retries exhausted)."""
    return patch("gim.scorers._call_exact_judge", new=AsyncMock(side_effect=exc))


def _mock_model(grader_model_value=None):
    """Patch get_model."""
    return patch("gim.scorers.get_model", return_value=MagicMock())


# ---------------------------------------------------------------------------
# _score_exact_answer tests
# ---------------------------------------------------------------------------


class TestScoreExactAnswer:
    async def test_correct_full_confidence(self):
        """CORRECT with confidence=1.0 → value=1.0."""
        state = _make_task_state("Q?", "Paris")
        target = Target(["Paris"])
        response = ExactAnswerJudgment(
            explanation="Matches", grade="CORRECT", confidence=1.0
        ).model_dump_json()

        with _mock_exact_judge(response), _mock_model():
            score = await _score_exact_answer(state, target, "mock-model")

        assert score.value == 1.0
        assert score.metadata["judge_type"] == "exact_answer"
        assert score.metadata["grade"] == "CORRECT"
        assert score.metadata["confidence"] == 1.0

    async def test_correct_partial_confidence(self):
        """CORRECT with confidence=0.7 → value=0.7."""
        state = _make_task_state("Q?", "Paris")
        target = Target(["Paris"])
        response = ExactAnswerJudgment(
            explanation="Likely matches", grade="CORRECT", confidence=0.7
        ).model_dump_json()

        with _mock_exact_judge(response), _mock_model():
            score = await _score_exact_answer(state, target, "mock-model")

        assert abs(score.value - 0.7) < 1e-9
        assert score.metadata["confidence"] == 0.7

    async def test_incorrect_always_zero(self):
        """INCORRECT → value=0.0 regardless of confidence."""
        state = _make_task_state("Q?", "London")
        target = Target(["Paris"])
        response = ExactAnswerJudgment(
            explanation="Wrong", grade="INCORRECT", confidence=0.9
        ).model_dump_json()

        with _mock_exact_judge(response), _mock_model():
            score = await _score_exact_answer(state, target, "mock-model")

        assert score.value == 0.0
        assert score.metadata["grade"] == "INCORRECT"
        assert score.metadata["confidence"] == 0.9

    async def test_no_golden_answer_skips(self):
        state = _make_task_state("Q?", "anything")
        target = Target([""])

        score = await _score_exact_answer(state, target, "mock-model")

        assert score.value == INCORRECT
        assert score.metadata["skipped"] is True

    async def test_judge_failure_after_retries_returns_zero(self):
        """When all retries fail, returns value=0.0 with parse_error flag."""
        state = _make_task_state("Q?", "answer")
        target = Target(["gold"])

        with _mock_exact_judge_raises(RuntimeError("model error")), _mock_model():
            score = await _score_exact_answer(state, target, "mock-model")

        assert score.value == 0.0
        assert score.metadata["parse_error"] is True
        assert score.metadata["confidence"] == 0.0

    async def test_answer_preserved_in_score(self):
        state = _make_task_state("Q?", "my answer")
        target = Target(["gold"])
        response = ExactAnswerJudgment(
            explanation="ok", grade="CORRECT", confidence=1.0
        ).model_dump_json()

        with _mock_exact_judge(response), _mock_model():
            score = await _score_exact_answer(state, target, "mock-model")

        assert score.answer == "my answer"


# ---------------------------------------------------------------------------
# _score_rubrics tests
# ---------------------------------------------------------------------------


class TestScoreRubrics:
    async def test_single_rubric_full_score(self):
        state = _make_task_state(
            "Q?",
            "Full response",
            metadata={"rubrics": ["Answers correctly"], "answer": "gold"},
        )
        target = Target(["gold"])
        response = RubricJudgment(
            explanation="Perfect", score=1.0, confidence=1.0
        ).model_dump_json()

        with _mock_rubric_judge(response), _mock_model():
            score = await _score_rubrics(state, target, "mock-model")

        assert score.value == 1.0
        assert score.metadata["judge_type"] == "rubrics"

    async def test_multiple_rubrics_averaged(self):
        """Two rubrics, both score=0.8 confidence=1.0 → mean=0.8."""
        state = _make_task_state(
            "Q?",
            "Partial response",
            metadata={"rubrics": ["Rubric A", "Rubric B"], "answer": "gold"},
        )
        target = Target(["gold"])
        response = RubricJudgment(
            explanation="Good", score=0.8, confidence=1.0
        ).model_dump_json()

        with _mock_rubric_judge(response), _mock_model():
            score = await _score_rubrics(state, target, "mock-model")

        assert abs(score.value - 0.8) < 1e-9

    async def test_confidence_weighting(self):
        """score=1.0, confidence=0.5 → weighted=0.5, mean=0.5."""
        state = _make_task_state(
            "Q?",
            "response",
            metadata={"rubrics": ["Single rubric"], "answer": "gold"},
        )
        target = Target(["gold"])
        response = RubricJudgment(
            explanation="ok", score=1.0, confidence=0.5
        ).model_dump_json()

        with _mock_rubric_judge(response), _mock_model():
            score = await _score_rubrics(state, target, "mock-model")

        assert abs(score.value - 0.5) < 1e-9
        assert abs(score.metadata["average_confidence"] - 0.5) < 1e-9

    async def test_mixed_rubric_aggregation(self):
        """(1.0*1.0 + 0.5*0.8) / 2 = 0.7"""
        state = _make_task_state(
            "Q?",
            "response",
            metadata={"rubrics": ["R1", "R2"], "answer": "gold"},
        )
        target = Target(["gold"])
        r1 = RubricJudgment(explanation="Perfect", score=1.0, confidence=1.0).model_dump_json()
        r2 = RubricJudgment(explanation="Partial", score=0.5, confidence=0.8).model_dump_json()
        judgments = [
            RubricJudgment.model_validate_json(r1),
            RubricJudgment.model_validate_json(r2),
        ]

        with _mock_rubric_judge_side_effect(judgments), _mock_model():
            score = await _score_rubrics(state, target, "mock-model")

        expected = (1.0 * 1.0 + 0.5 * 0.8) / 2
        assert abs(score.value - expected) < 1e-9

    async def test_no_rubrics_skips(self):
        state = _make_task_state("Q?", "response", metadata={"rubrics": []})
        target = Target(["gold"])

        score = await _score_rubrics(state, target, "mock-model")

        assert score.value == INCORRECT
        assert score.metadata["skipped"] is True

    async def test_missing_rubrics_key_skips(self):
        state = _make_task_state("Q?", "response", metadata={})
        target = Target(["gold"])

        score = await _score_rubrics(state, target, "mock-model")

        assert score.value == INCORRECT
        assert score.metadata["skipped"] is True

    async def test_judge_error_yields_zero_rubric(self):
        """When all retries fail for a rubric, that rubric contributes 0.0."""
        state = _make_task_state(
            "Q?",
            "response",
            metadata={"rubrics": ["Failing rubric"], "answer": "gold"},
        )
        target = Target(["gold"])

        with patch(
            "gim.scorers._call_rubric_judge", new=AsyncMock(side_effect=RuntimeError("fail"))
        ), _mock_model():
            score = await _score_rubrics(state, target, "mock-model")

        assert score.value == 0.0
        assert score.metadata["rubric_grades"][0]["score"] == 0.0
        assert score.metadata["rubric_grades"][0]["confidence"] == 0.0

    async def test_rubric_grades_metadata(self):
        state = _make_task_state(
            "Q?",
            "response",
            metadata={"rubrics": ["R1", "R2"], "answer": "gold"},
        )
        target = Target(["gold"])
        response = RubricJudgment(
            explanation="ok", score=0.7, confidence=0.9
        ).model_dump_json()

        with _mock_rubric_judge(response), _mock_model():
            score = await _score_rubrics(state, target, "mock-model")

        grades = score.metadata["rubric_grades"]
        assert len(grades) == 2
        assert all(g["rubric"] in ("R1", "R2") for g in grades)
        assert all(g["score"] == 0.7 for g in grades)

    async def test_falls_back_to_target_text_when_no_answer_metadata(self):
        """When metadata has no 'answer', uses target.text instead."""
        state = _make_task_state(
            "Q?",
            "response",
            metadata={"rubrics": ["Check answer"]},
        )
        target = Target(["fallback gold"])
        mock_judge = AsyncMock(
            return_value=RubricJudgment(explanation="ok", score=1.0, confidence=1.0)
        )

        with patch("gim.scorers._call_rubric_judge", new=mock_judge), _mock_model():
            await _score_rubrics(state, target, "mock-model")

        call_prompt = mock_judge.call_args[0][1]
        assert "fallback gold" in call_prompt


# ---------------------------------------------------------------------------
# Generation error handling
# ---------------------------------------------------------------------------


class TestGenerationErrorHandling:
    @pytest.mark.parametrize("completion", ["", "  \t\n"])
    async def test_blank_completion_is_missing(self, completion):
        """Empty and whitespace-only completions return a missing score."""
        state = _make_task_state("Q?", completion)
        target = Target(["gold"])

        scorer_fn = gim_scorer()
        score = await scorer_fn(state, target)

        assert math.isnan(score.as_float())
        assert score.metadata.get("generation_error") is True

    async def test_empty_completion_skips_judge(self):
        """No judge call is made when completion is empty."""
        state = _make_task_state("Q?", "", metadata={"rubrics": ["R1"]})
        target = Target(["gold"])
        mock_judge = AsyncMock()

        with patch("gim.scorers._call_rubric_judge", new=mock_judge):
            scorer_fn = gim_scorer()
            await scorer_fn(state, target)

        mock_judge.assert_not_called()

    async def test_empty_completion_excluded_from_repeat_mean(self):
        """NaN repeats are skipped; treating missing as zero would yield 0.3."""
        state = _make_task_state("Q?", "")
        missing = await gim_scorer()(state, Target(["gold"]))

        reduced = mean_score()([missing, Score(value=0.6)])

        assert reduced.as_float() == 0.6

    async def test_all_empty_completions_remain_missing_after_repeat_mean(self):
        """A sample with no successful repeats remains missing."""
        state = _make_task_state("Q?", "")
        missing = await gim_scorer()(state, Target(["gold"]))

        reduced = mean_score()([missing, missing])

        assert math.isnan(reduced.as_float())

    async def test_nonempty_completion_routes_normally(self):
        """Non-empty completion routes to the judge as normal."""
        state = _make_task_state("Q?", "Paris", metadata={})
        target = Target(["Paris"])
        response = ExactAnswerJudgment(
            explanation="Matches", grade="CORRECT", confidence=1.0
        ).model_dump_json()

        with _mock_exact_judge(response), _mock_model():
            scorer_fn = gim_scorer()
            score = await scorer_fn(state, target)

        assert score.value == 1.0
