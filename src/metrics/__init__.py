"""Metrics module for Lightspeed Core Stack."""

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
)

# Counter to track REST API calls
# This will be used to count how many times each API endpoint is called
# and the status code of the response
rest_api_calls_total = Counter(
    "ls_rest_api_calls_total", "REST API calls counter", ["path", "status_code"]
)

# Histogram to measure response durations
# This will be used to track how long it takes to handle requests
response_duration_seconds = Histogram(
    "ls_response_duration_seconds", "Response durations", ["path"]
)

# Metric that indicates what provider + model customers are using so we can
# understand what is popular/important
provider_model_configuration = Gauge(
    "ls_provider_model_configuration",
    "LLM provider/models combinations defined in configuration",
    ["provider", "model"],
)

# Metric that counts how many LLM calls were made for each provider + model
llm_calls_total = Counter(
    "ls_llm_calls_total", "LLM calls counter", ["provider", "model"]
)

# Metric that counts how many LLM calls failed
llm_calls_failures_total = Counter(
    "ls_llm_calls_failures_total", "LLM calls failures", ["provider", "model"]
)

# Metric that counts how many LLM calls had validation errors
llm_calls_validation_errors_total = Counter(
    "ls_llm_validation_errors_total", "LLM validation errors"
)

# Histogram to measure E2E LLM call durations per provider, model, and call type.
# Only used for non-streaming calls (query, rlsapi, topic_summary) where the full
# round-trip duration is meaningful.
llm_duration_seconds = Histogram(
    "ls_llm_duration_seconds",
    "LLM E2E call durations (non-streaming)",
    ["provider", "model", "call_type"],
)

# Histogram to measure time-to-first-token (TTFT) for streaming LLM calls.
# Captures the time from request creation until the stream is opened and the
# first chunk is received. Only used for streaming call types (streaming_query, a2a).
llm_ttft_seconds = Histogram(
    "ls_llm_ttft_seconds",
    "LLM time-to-first-token for streaming calls",
    ["provider", "model", "call_type"],
)

# Metric to count LLM tokens sent in requests, by provider, model, and call type.
llm_token_sent_total = Counter(
    "ls_llm_token_sent_total", "LLM tokens sent", ["provider", "model", "call_type"]
)

# Metric to count LLM tokens received in responses, by provider, model, and call type.
llm_token_received_total = Counter(
    "ls_llm_token_received_total",
    "LLM tokens received",
    ["provider", "model", "call_type"],
)

# Histogram to measure the total duration of streaming LLM calls.
llm_stream_duration_seconds = Histogram(
    "ls_llm_stream_duration_seconds",
    "LLM total stream duration for streaming calls",
    ["provider", "model", "call_type"],
)

# Histogram to record LLM-as-a-judge quality scores per metric dimension (0.0–1.0).
# Used for sampled interactions when llm_judge is enabled in configuration.
llm_judge_score = Histogram(
    "ls_llm_judge_score",
    "LLM-as-a-judge evaluation scores per metric dimension",
    ["provider", "model", "call_type", "metric_name"],
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
