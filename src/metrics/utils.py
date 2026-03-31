"""Utility functions for metrics handling."""

import time
from collections.abc import AsyncIterator
from typing import TypeVar

from fastapi import HTTPException
from llama_stack_client import APIConnectionError, APIStatusError
from prometheus_client import Histogram

import metrics
from client import AsyncLlamaStackClientHolder
from configuration import configuration
from log import get_logger
from models.responses import ServiceUnavailableResponse
from utils.common import run_once_async
from utils.endpoints import check_configuration_loaded

_T = TypeVar("_T")

logger = get_logger(__name__)


async def measure_stream_duration(
    start_time: float,
    histogram: Histogram,
    stream: AsyncIterator[_T],
) -> AsyncIterator[_T]:
    """Wrap an async stream to record total streaming duration.

    Observes the elapsed time from *start_time* until the last chunk is
    yielded from the stream, then re-yields all chunks unmodified.

    Args:
        start_time: Monotonic timestamp recorded immediately before the LLM
            call was issued (``time.monotonic()``).
        histogram: Prometheus Histogram instance (already labelled) to
            observe the stream duration value into.
        stream: The async iterator returned by the LLM streaming call.

    Yields:
        Chunks from the underlying stream, unmodified.
    """
    async for chunk in stream:
        yield chunk
    histogram.observe(time.monotonic() - start_time)


async def measure_ttft(
    start_time: float,
    histogram: Histogram,
    stream: AsyncIterator[_T],
) -> AsyncIterator[_T]:
    """
    wraps an async stream to measure the time-to-first-token (TTFT)

    As TTFT we measure the time from *start_time* (captured before the LLM
    call) until the first chunk is yielded from the stream, then re-yields
    all remaining chunks unmodified.

    Args:
        start_time: Monotonic timestamp recorded immediately before the LLM
            call was issued (``time.monotonic()``).
        histogram: Prometheus Histogram instance (already labelled) to
            observe the TTFT value into.
        stream: The async iterator returned by the LLM streaming call.

    Yields:
        Chunks from the underlying stream, unmodified.
    """
    first = True
    async for chunk in stream:
        if first:
            histogram.observe(time.monotonic() - start_time)
            first = False
        yield chunk


@run_once_async
async def setup_model_metrics() -> None:
    """Perform setup of all metrics related to LLM model and provider."""
    logger.info("Setting up model metrics")
    check_configuration_loaded(configuration)
    try:
        model_list = await AsyncLlamaStackClientHolder().get_client().models.list()
    except (APIConnectionError, APIStatusError) as e:
        response = ServiceUnavailableResponse(backend_name="Llama Stack", cause=str(e))
        raise HTTPException(**response.model_dump()) from e

    models = [
        model
        for model in model_list
        if model.custom_metadata and model.custom_metadata.get("model_type") == "llm"
    ]

    default_model_label = (
        configuration.inference.default_provider,  # type: ignore[reportAttributeAccessIssue]
        configuration.inference.default_model,  # type: ignore[reportAttributeAccessIssue]
    )

    for model in models:
        provider = (
            str(model.custom_metadata.get("provider_id", ""))
            if model.custom_metadata
            else ""
        )
        model_name = model.id
        if provider and model_name:
            # If the model/provider combination is the default, set the metric value to 1
            # Otherwise, set it to 0
            default_model_value = 0
            label_key = (provider, model_name)
            if label_key == default_model_label:
                default_model_value = 1

            # Set the metric for the provider/model configuration
            metrics.provider_model_configuration.labels(*label_key).set(
                default_model_value
            )
            logger.debug(
                "Set provider/model configuration for %s/%s to %d",
                provider,
                model_name,
                default_model_value,
            )
    logger.info("Model metrics setup complete")
