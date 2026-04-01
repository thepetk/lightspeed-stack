"""LLM-as-a-Judge evaluation utilities for scoring LLM interactions across quality metrics."""

import asyncio
import json
import random
import re
from typing import Optional

from llama_stack_client import AsyncLlamaStackClient

import metrics
from constants import (
    LLM_JUDGE_ALL_METRICS,
    LLM_JUDGE_CRITERIA,
    LLM_JUDGE_METRIC_ANSWER_RELEVANCY,
    LLM_JUDGE_METRIC_BIAS,
    LLM_JUDGE_METRIC_CONTEXTUAL_RELEVANCY,
    LLM_JUDGE_METRIC_HALLUCINATION,
    LLM_JUDGE_METRIC_HELPFULNESS,
    LLM_JUDGE_METRIC_PROMPT_ALIGNMENT,
    LLM_JUDGE_METRIC_TOXICITY,
    LLM_JUDGE_SCORING_SYSTEM_PROMPT,
    LLM_JUDGE_STEP_GEN_SYSTEM_PROMPT,
)
from log import get_logger
from utils.query import extract_provider_and_model_from_model_id
from utils.responses import extract_text_from_response_items

logger = get_logger(__name__)

# Module-level aliases — convenience re-exports used by tests and callers.
ALL_METRICS = LLM_JUDGE_ALL_METRICS
METRIC_ANSWER_RELEVANCY = LLM_JUDGE_METRIC_ANSWER_RELEVANCY
METRIC_CONTEXTUAL_RELEVANCY = LLM_JUDGE_METRIC_CONTEXTUAL_RELEVANCY
METRIC_TOXICITY = LLM_JUDGE_METRIC_TOXICITY
METRIC_BIAS = LLM_JUDGE_METRIC_BIAS
METRIC_PROMPT_ALIGNMENT = LLM_JUDGE_METRIC_PROMPT_ALIGNMENT
METRIC_HELPFULNESS = LLM_JUDGE_METRIC_HELPFULNESS
METRIC_HALLUCINATION = LLM_JUDGE_METRIC_HALLUCINATION


def should_sample(sampling_rate: float) -> bool:
    """Return True if this interaction should be evaluated by the judge.

    Args:
        sampling_rate: Probability in [0.0, 1.0] that any given
            interaction is evaluated.

    Returns:
        bool: True when a uniform random draw falls below sampling_rate.
    """
    return random.random() < sampling_rate


async def _generate_evaluation_steps(
    metric_name: str,
    client: AsyncLlamaStackClient,
    judge_model: str,
) -> str:
    """Generate evaluation steps for a metric using the judge LLM (G-Eval phase 1).

    Asks the judge to produce a numbered list of concrete evaluation steps
    derived from the task description and criteria for the given metric.

    Args:
        metric_name: One of the ALL_METRICS strings.
        client: The AsyncLlamaStackClient instance.
        judge_model: Full model ID of the judge LLM.

    Returns:
        str: Numbered evaluation steps as plain text, or ``""`` on failure.
    """
    task, criteria = LLM_JUDGE_CRITERIA[metric_name]
    user_prompt = (
        f"Task: {task}\n\nCriteria: {criteria}\n\nGenerate evaluation steps:"
    )
    try:
        response = await client.responses.create(
            input=user_prompt,
            model=judge_model,
            instructions=LLM_JUDGE_STEP_GEN_SYSTEM_PROMPT,
            stream=False,
            store=False,
        )
        return extract_text_from_response_items(response.output)
    except Exception:  # pylint: disable=broad-except
        logger.warning(
            "LLM judge: failed to generate evaluation steps for metric '%s'",
            metric_name,
        )
        return ""


def _build_judge_user_prompt(
    metric_name: str,
    query: str,
    response: str,
    rag_context: str,
    system_prompt: str,
    evaluation_steps: str = "",
) -> str:
    """Construct the user-facing scoring prompt for the judge (G-Eval phase 2).

    Args:
        metric_name: The metric being evaluated (e.g. "answer_relevancy").
        query: The user's original question.
        response: The assistant's response.
        rag_context: Retrieved context chunks joined as a single string.
        system_prompt: The system prompt used for this interaction.
        evaluation_steps: Numbered steps generated in phase 1. When non-empty
            they are injected before the scoring instruction.

    Returns:
        str: The formatted user prompt.
    """
    parts = []
    if evaluation_steps:
        parts.append(f"## Evaluation Steps\n{evaluation_steps}\n")
    parts.append(f"## User Question\n{query}\n")
    if system_prompt:
        parts.append(f"## System Prompt\n{system_prompt}\n")
    if rag_context:
        parts.append(f"## Retrieved Context\n{rag_context}\n")
    parts.append(f"## Assistant Response\n{response}\n")
    parts.append(
        f"Follow the evaluation steps above and score the "
        f"**{metric_name.replace('_', ' ')}** of the assistant response "
        f"from 1 (worst) to 5 (best)."
    )
    return "\n".join(parts)


def _parse_score(raw: str) -> Optional[float]:
    """Extract a 1–5 integer score and normalize it to [0.0, 1.0].

    Strategy:
    1. Try to parse JSON ``{"score": N}`` where ``1 <= N <= 5``.
    2. Fall back to the first bare integer in range [1-5] found by regex.
    3. Normalize: ``(score - 1) / 4.0``.

    Args:
        raw: The raw text response from the judge model.

    Returns:
        Optional[float]: Normalized score in [0.0, 1.0], or None on failure.
    """
    json_match = re.search(r'\{[^{}]*"score"\s*:\s*(-?[\d.]+)[^{}]*\}', raw)
    if json_match:
        try:
            obj = json.loads(json_match.group(0))
            score = float(obj["score"])
            if 1.0 <= score <= 5.0:
                return (score - 1.0) / 4.0
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

    int_match = re.search(r"\b([1-5])\b", raw)
    if int_match:
        try:
            score = float(int_match.group(1))
            return (score - 1.0) / 4.0
        except ValueError:
            pass

    return None


async def _evaluate_single_metric(
    metric_name: str,
    query: str,
    response: str,
    rag_context: str,
    system_prompt: str,
    model_id: str,
    call_type: str,
    judge_model: str,
    client: AsyncLlamaStackClient,
) -> None:
    """Run the two-phase G-Eval judge for a single metric and observe the result.

    Phase 1 generates evaluation steps; phase 2 scores the interaction using
    those steps. Any exception is caught and logged so that one failing metric
    does not abort the others or propagate to the user.

    Args:
        metric_name: One of the ALL_METRICS strings.
        query: User's original question.
        response: Assistant's response.
        rag_context: Retrieved context joined as a string.
        system_prompt: The system prompt used for this interaction.
        model_id: Full model ID (provider/model) of the inference model.
        call_type: "query" or "streaming_query".
        judge_model: Full model ID of the judge LLM.
        client: The AsyncLlamaStackClient instance.
    """
    # generate evaluation steps from task + criteria
    evaluation_steps = await _generate_evaluation_steps(
        metric_name, client, judge_model
    )

    # score using the generated steps
    user_prompt = _build_judge_user_prompt(
        metric_name, query, response, rag_context, system_prompt, evaluation_steps
    )

    try:
        judge_response = await client.responses.create(
            input=user_prompt,
            model=judge_model,
            instructions=LLM_JUDGE_SCORING_SYSTEM_PROMPT,
            stream=False,
            store=False,
        )
        raw_text = extract_text_from_response_items(judge_response.output)
        score = _parse_score(raw_text)
        if score is None:
            logger.warning(
                "LLM judge: could not parse score for metric '%s' from response: %r",
                metric_name,
                raw_text[:200],
            )
            return

        provider, model_label = extract_provider_and_model_from_model_id(model_id)
        metrics.llm_judge_score.labels(
            provider=provider,
            model=model_label,
            call_type=call_type,
            metric_name=metric_name,
        ).observe(score)

        logger.debug(
            "LLM judge: metric=%s score=%.3f provider=%s model=%s call_type=%s",
            metric_name,
            score,
            provider,
            model_label,
            call_type,
        )

    except Exception:  # pylint: disable=broad-except
        logger.exception("LLM judge: error evaluating metric '%s'", metric_name)


async def evaluate_interaction(
    query: str,
    response: str,
    rag_chunks: list,
    system_prompt: str,
    model_id: str,
    call_type: str,
    judge_model: str,
    client: AsyncLlamaStackClient,
) -> None:
    """Evaluate a completed LLM interaction across all seven quality metrics.

    Each metric runs through the two-phase G-Eval flow independently and
    concurrently via asyncio.gather(). Any individual failure is logged and
    suppressed so that one bad judge call does not abort the others.

    Args:
        query: The user's original question.
        response: The assistant's full response.
        rag_chunks: List of RAGChunk objects used for retrieval (may be empty).
        system_prompt: The system prompt that governed this interaction.
        model_id: Full ID (provider/model) of the inference model.
        call_type: "query" or "streaming_query".
        judge_model: Full model ID of the judge LLM.
        client: The shared AsyncLlamaStackClient.
    """
    rag_context = "\n\n".join(
        chunk.content for chunk in rag_chunks if getattr(chunk, "content", None)
    )

    tasks = [
        _evaluate_single_metric(
            metric_name=metric,
            query=query,
            response=response,
            rag_context=rag_context,
            system_prompt=system_prompt,
            model_id=model_id,
            call_type=call_type,
            judge_model=judge_model,
            client=client,
        )
        for metric in LLM_JUDGE_ALL_METRICS
    ]

    await asyncio.gather(*tasks, return_exceptions=True)
