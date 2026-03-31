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
    LLM_JUDGE_SYSTEM_PROMPTS,
)
from log import get_logger
from utils.query import extract_provider_and_model_from_model_id
from utils.responses import extract_text_from_response_items

logger = get_logger(__name__)


def should_sample(sampling_rate: float) -> bool:
    """Return True if this interaction should be evaluated by the judge.

    Args:
        sampling_rate: Probability in [0.0, 1.0] that any given
            interaction is evaluated.

    Returns:
        bool: True when a uniform random draw falls below sampling_rate.
    """
    return random.random() < sampling_rate


def _build_judge_user_prompt(
    metric_name: str,
    query: str,
    response: str,
    rag_context: str,
    system_prompt: str,
) -> str:
    """Construct the user-facing evaluation prompt for the judge.

    Args:
        metric_name: The metric being evaluated (e.g. "answer_relevancy").
        query: The user's original question.
        response: The assistant's response.
        rag_context: Retrieved context chunks joined as a single string.
        system_prompt: The system prompt used for this interaction.

    Returns:
        str: The formatted user prompt.
    """
    parts = [f"## User Question\n{query}\n"]
    if system_prompt:
        parts.append(f"## System Prompt\n{system_prompt}\n")
    if rag_context:
        parts.append(f"## Retrieved Context\n{rag_context}\n")
    parts.append(f"## Assistant Response\n{response}\n")
    parts.append(
        f"Evaluate the **{metric_name.replace('_', ' ')}** of the assistant response."
    )
    return "\n".join(parts)


def _parse_score(raw: str) -> Optional[float]:
    """Extract a 0.0–1.0 float score from a judge response string.

    Strategy:
    1. Try to parse JSON ``{"score": <float>}`` from the response.
    2. Fall back to the first bare float found by regex.
    3. Clamp result to [0.0, 1.0].

    Args:
        raw: The raw text response from the judge model.

    Returns:
        Optional[float]: Parsed score in [0.0, 1.0], or None on failure.
    """
    # JSON extraction — handles preamble/postamble around the object
    json_match = re.search(r'\{[^{}]*"score"\s*:\s*(-?[\d.]+)[^{}]*\}', raw)
    if json_match:
        try:
            obj = json.loads(json_match.group(0))
            score = float(obj["score"])
            return max(0.0, min(1.0, score))
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

    # Bare float fallback
    float_match = re.search(r"\b(0(?:\.\d+)?|1(?:\.0+)?)\b", raw)
    if float_match:
        try:
            score = float(float_match.group(1))
            return max(0.0, min(1.0, score))
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
    """Run the judge for a single metric and observe the result.

    Any exception is caught and logged so that one failing metric does not
    abort the others or propagate to the user.

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
    system_instruction = LLM_JUDGE_SYSTEM_PROMPTS[metric_name]
    user_prompt = _build_judge_user_prompt(
        metric_name, query, response, rag_context, system_prompt
    )

    try:
        judge_response = await client.responses.create(
            input=user_prompt,
            model=judge_model,
            instructions=system_instruction,
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

    Each metric is evaluated by an independent judge LLM call. All calls run
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
