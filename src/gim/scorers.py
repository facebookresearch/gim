# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""GIM scorers: exact-answer and rubric-graded LLM-as-judge.

Two scoring strategies, selected per-sample based on available metadata:

1. **Rubric-graded** (preferred): Each rubric item is evaluated independently
   by a grader model. Scores are confidence-weighted and averaged:
   final_score = sum(score_i * confidence_i) / n_rubrics
2. **Exact-answer**: A grader model compares the model output against a golden
   answer, returning CORRECT / INCORRECT plus a confidence score.
   Score = confidence if CORRECT, 0.0 if INCORRECT.

When a sample has rubrics, the rubric score is used. Otherwise the exact-answer
score is used. This mirrors the priority logic in the TBR implementation.

Failed generations (empty completion) are recorded as NaN so aggregate metrics
and epoch reducers treat them as missing observations.

All judge calls are retried up to 15 times with exponential backoff via stamina.

Diagnostic metrics:
- solved: 1.0 if generation succeeded, 0.0 if it failed
- scored: 1.0 if judging succeeded, 0.0 if generation failed
- scored/grader/{judge_type}: 1.0 per successful grading by type
- solved/error/{error_type}: 1.0 when generation fails, classified by cause
- reward/modality/{tag}: reward keyed by modality for per-modality aggregation
"""

import asyncio
import logging
from typing import Any, Literal

import stamina
from inspect_ai.model import ChatMessageUser, GenerateConfig, get_model, ResponseSchema
from inspect_ai.scorer import INCORRECT, Score, Scorer, scorer, stderr, Target
from inspect_ai.solver import TaskState
from inspect_ai.util._json import cls_json_schema
from pydantic import BaseModel, Field

from .judges import judge_metadata, resolve_judge
from .metrics import gim_per_modality, gim_score, raw_mean
from .modality import modality_supersets

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response schemas for structured output
# ---------------------------------------------------------------------------


class ExactAnswerJudgment(BaseModel):
    """Judge response for exact-answer comparison."""

    explanation: str = Field(description="Explanation of how the grade was determined.")
    grade: Literal["CORRECT", "INCORRECT"] = Field(
        description="Whether the predicted answer matches the gold target."
    )
    confidence: float = Field(
        description="Reliability of the grade, from 0.0 (least reliable) to 1.0 (most reliable).",
        ge=0.0,
        le=1.0,
    )


class RubricJudgment(BaseModel):
    """Judge response for rubric-graded evaluation."""

    explanation: str = Field(
        description="Explanation of how the score was determined by applying the rubric."
    )
    score: float = Field(
        description="How well the response meets the rubric criteria, from 0.0 (not met) to 1.0 (fully met).",
        ge=0.0,
        le=1.0,
    )
    confidence: float = Field(
        description="Reliability of the score, from 0.0 (least reliable) to 1.0 (most reliable).",
        ge=0.0,
        le=1.0,
    )


EXACT_ANSWER_SCHEMA = ResponseSchema(
    name="exact_answer_judgment",
    description="Judge whether a predicted answer matches the gold target.",
    json_schema=cls_json_schema(ExactAnswerJudgment),
)

RUBRIC_SCHEMA = ResponseSchema(
    name="rubric_judgment",
    description="Evaluate a model response against a rubric item.",
    json_schema=cls_json_schema(RubricJudgment),
)


# ---------------------------------------------------------------------------
# Judge prompts
# ---------------------------------------------------------------------------

# Adapted from https://github.com/openai/simple-evals/blob/main/simpleqa_eval.py
EXACT_ANSWER_PROMPT = """\
Task Description
I have three inputs for you:
    1) Question: The first input is the question.
    2) Gold target: The second input is the expected answer.
    3) Predicted answer: The third input is the answer to verify, which may \
contain reasoning and should end with a specific format (e.g., \
"Final answer: the final answer is", "Answer:", or "Answer is").

Task Requirements
Your task is to:
    1) Check if a real answer is generated.
    2) If it does, extract the final answer and compare it with the golden target.
    3) Consider answers as the same if they are represented in different \
formats (e.g., 0.5 and 1/2) or have an absolute difference of less than \
0.01 (e.g., sqrt(2) and 1.41).
    4) Handle non-numeric answers, such as booleans or lists of strings, \
where order matters.
    5) Account for choice questions where the expected answer is a letter \
(A, B, C, D) and the real answer is a string mentioned in the question.
    6) Treat N/A as equivalent to null, none, or empty in program analysis \
outputs.
    7) Identify answers that involve context, such as "the answer is \
something shown above".

Here is a new example. Grade the predicted answer as one of: CORRECT, INCORRECT.
```
Question: {question}
Gold target: {answer}
Predicted answer: {predicted_answer}
```

The gold target is the ground truth—do not question its correctness. \
Even if the predicted answer appears more reasonable or accurate than the \
gold target, grade it as INCORRECT if it does not match the gold target.

First provide the explanation, then grade the predicted answer.

# Confidence

Also provide a confidence value between 0.0 and 1.0 indicating how \
reliable your score is. 1.0 means you are fully certain; 0.0 means \
the score is a guess."""


RUBRIC_GRADER_PROMPT = """\
You are a helpful grader, and will be given a conversation including a \
prompt, expected golden answer, a model response and a rubric item.
Your job is to look at the conversation and the rubric string, and \
determine whether the response meets the criteria specified in the rubric \
string.
Please note that the rubric may ask the information from either prompt, \
golden response or model response, and itself may contain some ground truth.

# Golden response
{answer}

# Model Response
{model_response}

# Rubric string
{rubric_string}

When you finish your evaluation, you'll need to provide a final score as \
a float between 0.0 and 1.0:

# Scoring

- 1.0 means the rubric criterion is fully met.
- 0.0 means the criterion is not met at all.
- For partial credit, use a proportional value (e.g., 0.6 for 6 out of 10).
- If you don't have enough information to make a judgment, return 0.0.

# Confidence

Also provide a confidence value between 0.0 and 1.0 indicating how \
reliable your score is. 1.0 means you are fully certain; 0.0 means \
the score is a guess.

# Important Note on Criteria Examples

When a criterion includes phrases like "such as," "for example," or \
"including," the response doesn't need to cover every example listed to \
meet the criterion. For instance, if a criterion states, "States that oral \
iron supplements can lead to unpleasant gastrointestinal side effects such \
as nausea, vomiting, and constipation," and the response only mentions \
"oral iron supplements can lead to unpleasant gastrointestinal side effects \
such as cramps," it still satisfies the criterion."""


# ---------------------------------------------------------------------------
# Retried judge helpers
# ---------------------------------------------------------------------------


@stamina.retry(on=Exception, attempts=15, wait_max=180.0)
async def _call_exact_judge(model, prompt: str) -> ExactAnswerJudgment:
    """Call the exact-answer judge, retrying up to 15 times on any failure."""
    result = await model.generate(
        [ChatMessageUser(content=prompt)],
        config=GenerateConfig(temperature=0, response_schema=EXACT_ANSWER_SCHEMA),
    )
    return ExactAnswerJudgment.model_validate_json(result.completion)


@stamina.retry(on=Exception, attempts=15, wait_max=180.0)
async def _call_rubric_judge(model, prompt: str) -> RubricJudgment:
    """Call the rubric judge, retrying up to 15 times on any failure."""
    result = await model.generate(
        [ChatMessageUser(content=prompt)],
        config=GenerateConfig(temperature=0, response_schema=RUBRIC_SCHEMA),
    )
    return RubricJudgment.model_validate_json(result.completion)


# ---------------------------------------------------------------------------
# Diagnostic metadata helpers
# ---------------------------------------------------------------------------


def _diagnostic_metadata(
    state: TaskState,
    judge_type: str,
    reward: float,
) -> dict[str, Any]:
    """Build diagnostic metadata matching TBR's sampled/graded counters."""
    tag = state.metadata.get("modality", "unknown")
    meta: dict[str, Any] = {
        "solved": 1.0,
        "scored": 1.0,
        f"scored/grader/{judge_type}": 1.0,
        f"solved/modality/{tag}": 1.0,
        f"scored/modality/{tag}": 1.0,
        f"reward/modality/{tag}": reward,
    }
    for superset in modality_supersets(tag):
        meta[f"reward/modality/{superset}"] = reward
    return meta


def _generation_failure_metadata(state: TaskState) -> dict[str, Any]:
    """Build diagnostic metadata for a failed generation."""
    tag = state.metadata.get("modality", "unknown")
    error_type = state.metadata.get("generation_error_type", "unknown")
    meta: dict[str, Any] = {
        "generation_error": True,
        "solved": 0.0,
        "scored": 0.0,
        f"solved/modality/{tag}": 0.0,
        f"solved/error/{error_type}": 1.0,
        f"reward/modality/{tag}": 0.0,
    }
    for superset in modality_supersets(tag):
        meta[f"reward/modality/{superset}"] = 0.0
    return meta


# ---------------------------------------------------------------------------
# Exact-answer scorer
# ---------------------------------------------------------------------------


async def _score_exact_answer(
    state: TaskState,
    target: Target,
    judge_model: str | None,
    judge_meta: dict[str, str] | None = None,
) -> Score:
    """Score a sample by comparing model output to the golden answer."""
    answer = target.text
    if not answer:
        return Score(
            value=INCORRECT,
            answer=state.output.completion,
            explanation="No golden answer available — skipped.",
            metadata={
                "judge_type": "exact_answer",
                "skipped": True,
                **(judge_meta or {}),
                **_diagnostic_metadata(state, "exact_answer", 0.0),
            },
        )

    prompt = EXACT_ANSWER_PROMPT.format(
        question=state.input_text,
        predicted_answer=state.output.completion,
        answer=answer,
    )

    model = get_model(judge_model)
    try:
        judgment = await _call_exact_judge(model, prompt)
    except Exception as exc:
        logger.warning("GIMExactAnswerGrader failed after all retries: %s", exc)
        return Score(
            value=0.0,
            answer=state.output.completion,
            explanation=f"Judge failed after retries: {exc}",
            metadata={
                "judge_type": "exact_answer",
                "parse_error": True,
                "confidence": 0.0,
                **(judge_meta or {}),
                **_diagnostic_metadata(state, "exact_answer", 0.0),
            },
        )

    # Confidence-weighted score: CORRECT → confidence, INCORRECT → 0.0
    value = judgment.confidence if judgment.grade == "CORRECT" else 0.0

    return Score(
        value=value,
        answer=state.output.completion,
        explanation=judgment.explanation,
        metadata={
            "judge_type": "exact_answer",
            "grade": judgment.grade,
            "confidence": judgment.confidence,
            **(judge_meta or {}),
            **_diagnostic_metadata(state, "exact_answer", value),
        },
    )


# ---------------------------------------------------------------------------
# Rubric-graded scorer
# ---------------------------------------------------------------------------


async def _score_rubrics(
    state: TaskState,
    target: Target,
    judge_model: str | None,
    judge_meta: dict[str, str] | None = None,
) -> Score:
    """Score a sample by evaluating each rubric item independently."""
    rubrics: list[str] = state.metadata.get("rubrics", [])
    if not rubrics:
        return Score(
            value=INCORRECT,
            answer=state.output.completion,
            explanation="No rubrics available — skipped.",
            metadata={
                "judge_type": "rubrics",
                "skipped": True,
                **(judge_meta or {}),
                **_diagnostic_metadata(state, "rubrics", 0.0),
            },
        )

    answer = state.metadata.get("answer", target.text or "")
    model_response = state.output.completion
    model = get_model(judge_model)

    async def _grade_one_rubric(rubric: str) -> dict[str, Any]:
        prompt = RUBRIC_GRADER_PROMPT.format(
            answer=answer,
            model_response=model_response,
            rubric_string=rubric,
        )
        try:
            judgment = await _call_rubric_judge(model, prompt)
            return {
                "rubric": rubric,
                "score": judgment.score,
                "confidence": judgment.confidence,
                "explanation": judgment.explanation,
            }
        except Exception as exc:
            logger.error(
                "Rubric judge failed for '%s' after retries: %s", rubric[:50], exc
            )
            return {
                "rubric": rubric,
                "score": 0.0,
                "confidence": 0.0,
                "explanation": f"Judge error: {exc}",
            }

    rubric_grades = list(await asyncio.gather(*[_grade_one_rubric(r) for r in rubrics]))

    # Confidence-weighted aggregation:
    # weighted_score = score * confidence (both in [0, 1])
    # Final score = mean of weighted scores across rubrics
    n = len(rubric_grades)
    total_weighted = sum(rg["score"] * rg["confidence"] for rg in rubric_grades)
    aggregate_score = total_weighted / n if n > 0 else 0.0
    avg_confidence = sum(rg["confidence"] for rg in rubric_grades) / n if n > 0 else 0.0

    explanations = [
        f"[{rg['rubric'][:60]}] score={rg['score']}, conf={rg['confidence']}: "
        f"{rg['explanation'][:100]}"
        for rg in rubric_grades
    ]

    return Score(
        value=aggregate_score,
        answer=state.output.completion,
        explanation="\n".join(explanations),
        metadata={
            "judge_type": "rubrics",
            "rubric_grades": rubric_grades,
            "average_confidence": avg_confidence,
            **(judge_meta or {}),
            **_diagnostic_metadata(state, "rubrics", aggregate_score),
        },
    )


# ---------------------------------------------------------------------------
# Composite GIM scorer
# ---------------------------------------------------------------------------


@scorer(metrics=[gim_score(), raw_mean(), gim_per_modality(), stderr()])
def gim_scorer(
    judge_id: str | None = None,
    judge_model: str | None = None,
) -> Scorer:
    """GIM composite scorer.

    Routes each sample to the appropriate judging strategy:
    - If generation failed (empty completion), returns NaN immediately
      with diagnostic metadata so aggregate metrics treat it as missing.
    - If rubrics are available, uses rubric-graded scoring (preferred).
    - Otherwise, uses exact-answer scoring against the golden answer.

    Args:
        judge_id: Canonical calibrated judge ID used for official IRT scoring.
            Inferred from ``judge_model`` when it matches a calibrated route.
        judge_model: Provider/model route used to call the judge.
    """
    judge_spec, resolved_judge_model = resolve_judge(judge_id, judge_model)
    resolved_judge_meta = judge_metadata(judge_spec, resolved_judge_model)

    async def score(state: TaskState, target: Target) -> Score:
        if not state.output.completion.strip():
            return Score(
                value=float("nan"),
                explanation="Generation produced no output — treated as missing.",
                metadata={
                    **resolved_judge_meta,
                    **_generation_failure_metadata(state),
                },
            )
        rubrics = state.metadata.get("rubrics", [])
        if rubrics:
            return await _score_rubrics(
                state, target, resolved_judge_model, resolved_judge_meta
            )
        else:
            return await _score_exact_answer(
                state, target, resolved_judge_model, resolved_judge_meta
            )

    return score
