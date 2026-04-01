"""Unit tests for utils/llm_judge.py."""

import pytest
from pytest_mock import MockerFixture

from utils.llm_judge import (
    ALL_METRICS,
    METRIC_ANSWER_RELEVANCY,
    METRIC_BIAS,
    METRIC_CONTEXTUAL_RELEVANCY,
    METRIC_HALLUCINATION,
    METRIC_HELPFULNESS,
    METRIC_PROMPT_ALIGNMENT,
    METRIC_TOXICITY,
    _build_judge_user_prompt,
    _evaluate_single_metric,
    _generate_evaluation_steps,
    _parse_score,
    evaluate_interaction,
    should_sample,
)


@pytest.fixture(name="mock_client")
def mock_client_fixture(mocker: MockerFixture):
    """Return a mock AsyncLlamaStackClient with responses.create stubbed."""
    client = mocker.MagicMock()
    client.responses.create = mocker.AsyncMock()
    return client


def _make_response_object(text: str, mocker: MockerFixture):
    """Build a minimal mock response object whose .output yields the given text."""
    output_item = mocker.MagicMock()
    output_item.type = "message"
    content_part = mocker.MagicMock()
    content_part.type = "output_text"
    content_part.text = text
    output_item.content = [content_part]
    response = mocker.MagicMock()
    response.output = [output_item]
    return response


class TestShouldSample:
    """Tests for should_sample()."""

    def test_always_samples_at_rate_one(self, mocker: MockerFixture) -> None:
        """sampling_rate=1.0 always returns True regardless of random draw."""
        mocker.patch("utils.llm_judge.random.random", return_value=0.0)
        assert should_sample(1.0) is True

    def test_never_samples_at_rate_zero(self) -> None:
        """sampling_rate=0.0 always returns False."""
        assert should_sample(0.0) is False

    def test_samples_when_draw_is_below_rate(self, mocker: MockerFixture) -> None:
        """Returns True when random draw < sampling_rate."""
        mocker.patch("utils.llm_judge.random.random", return_value=0.49)
        assert should_sample(0.5) is True

    def test_does_not_sample_when_draw_equals_rate(self, mocker: MockerFixture) -> None:
        """Returns False when random draw == sampling_rate (boundary miss)."""
        mocker.patch("utils.llm_judge.random.random", return_value=0.5)
        assert should_sample(0.5) is False

    def test_does_not_sample_when_draw_is_above_rate(
        self, mocker: MockerFixture
    ) -> None:
        """Returns False when random draw > sampling_rate."""
        mocker.patch("utils.llm_judge.random.random", return_value=0.9)
        assert should_sample(0.1) is False


class TestParseScore:
    """Tests for _parse_score().

    Scores are expected on a 1-5 integer scale, normalized to 0.0-1.0:
      1 -> 0.0, 2 -> 0.25, 3 -> 0.5, 4 -> 0.75, 5 -> 1.0
    """

    def test_score_1_normalizes_to_0(self) -> None:
        """Score 1 normalizes to 0.0."""
        assert _parse_score('{"score": 1}') == pytest.approx(0.0)

    def test_score_2_normalizes_to_0_25(self) -> None:
        """Score 2 normalizes to 0.25."""
        assert _parse_score('{"score": 2}') == pytest.approx(0.25)

    def test_score_3_normalizes_to_0_5(self) -> None:
        """Score 3 normalizes to 0.5."""
        assert _parse_score('{"score": 3}') == pytest.approx(0.5)

    def test_score_4_normalizes_to_0_75(self) -> None:
        """Score 4 normalizes to 0.75."""
        assert _parse_score('{"score": 4}') == pytest.approx(0.75)

    def test_score_5_normalizes_to_1(self) -> None:
        """Score 5 normalizes to 1.0."""
        assert _parse_score('{"score": 5}') == pytest.approx(1.0)

    def test_json_with_preamble(self) -> None:
        """Parses score from JSON embedded in surrounding text."""
        assert _parse_score(
            'Here is my evaluation: {"score": 4} done.'
        ) == pytest.approx(0.75)

    def test_bare_integer_fallback(self) -> None:
        """Falls back to bare integer when no JSON is present."""
        assert _parse_score("3") == pytest.approx(0.5)

    def test_out_of_range_score_0_returns_none(self) -> None:
        """Score 0 is out of 1-5 range and returns None."""
        assert _parse_score('{"score": 0}') is None

    def test_out_of_range_score_6_returns_none(self) -> None:
        """Score 6 is out of 1-5 range and returns None."""
        assert _parse_score('{"score": 6}') is None

    def test_unparseable_returns_none(self) -> None:
        """Returns None when response contains no recognisable score."""
        assert _parse_score("I cannot evaluate this.") is None

    def test_malformed_json_falls_back_to_regex(self) -> None:
        """Falls back to bare integer regex when JSON is malformed."""
        # "abc" is not a number — but there is no bare 1-5 integer either
        assert _parse_score('{"score": abc}') is None

    def test_malformed_json_with_recoverable_integer(self) -> None:
        """Bare integer found after malformed JSON is used as fallback."""
        assert _parse_score('{"score": abc} rating: 3') == pytest.approx(0.5)


class TestBuildJudgeUserPrompt:
    """Tests for _build_judge_user_prompt()."""

    def test_metric_name_appears_in_output(self) -> None:
        """The metric name (with spaces) appears in the scoring instruction."""
        prompt = _build_judge_user_prompt("answer_relevancy", "q", "r", "", "")
        assert "answer relevancy" in prompt

    def test_system_prompt_included_when_non_empty(self) -> None:
        """## System Prompt section is present when system_prompt is non-empty."""
        prompt = _build_judge_user_prompt(
            METRIC_HELPFULNESS, "q", "r", "", "Be concise."
        )
        assert "## System Prompt" in prompt
        assert "Be concise." in prompt

    def test_system_prompt_absent_when_empty(self) -> None:
        """## System Prompt section is absent when system_prompt is empty."""
        prompt = _build_judge_user_prompt(METRIC_HELPFULNESS, "q", "r", "", "")
        assert "## System Prompt" not in prompt

    def test_rag_context_included_when_non_empty(self) -> None:
        """## Retrieved Context section is present when rag_context is non-empty."""
        prompt = _build_judge_user_prompt(
            METRIC_CONTEXTUAL_RELEVANCY, "q", "r", "some docs", ""
        )
        assert "## Retrieved Context" in prompt
        assert "some docs" in prompt

    def test_rag_context_absent_when_empty(self) -> None:
        """## Retrieved Context section is absent when rag_context is empty."""
        prompt = _build_judge_user_prompt(METRIC_CONTEXTUAL_RELEVANCY, "q", "r", "", "")
        assert "## Retrieved Context" not in prompt

    def test_query_and_response_always_present(self) -> None:
        """User question and assistant response are always included."""
        prompt = _build_judge_user_prompt(
            METRIC_TOXICITY, "What is Python?", "Python is a language.", "", ""
        )
        assert "What is Python?" in prompt
        assert "Python is a language." in prompt

    def test_evaluation_steps_included_when_provided(self) -> None:
        """## Evaluation Steps section is present when evaluation_steps is non-empty."""
        steps = "1. Check relevance\n2. Check completeness"
        prompt = _build_judge_user_prompt(
            METRIC_ANSWER_RELEVANCY, "q", "r", "", "", evaluation_steps=steps
        )
        assert "## Evaluation Steps" in prompt
        assert steps in prompt

    def test_evaluation_steps_absent_by_default(self) -> None:
        """## Evaluation Steps section is absent when evaluation_steps is empty."""
        prompt = _build_judge_user_prompt(METRIC_ANSWER_RELEVANCY, "q", "r", "", "")
        assert "## Evaluation Steps" not in prompt


class TestGenerateEvaluationSteps:
    """Tests for _generate_evaluation_steps()."""

    @pytest.mark.asyncio
    async def test_success_returns_steps_text(
        self, mock_client, mocker: MockerFixture
    ) -> None:
        """Returns the text from the judge response on success."""
        mock_client.responses.create.return_value = _make_response_object(
            "1. Check A\n2. Check B", mocker
        )

        result = await _generate_evaluation_steps(
            METRIC_ANSWER_RELEVANCY, mock_client, "judge/model"
        )

        assert result == "1. Check A\n2. Check B"
        mock_client.responses.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_exception_returns_empty_string(self, mock_client) -> None:
        """Returns '' without propagating when the LLM call raises."""
        mock_client.responses.create.side_effect = RuntimeError("timeout")

        result = await _generate_evaluation_steps(
            METRIC_HALLUCINATION, mock_client, "judge/model"
        )

        assert result == ""


class TestEvaluateSingleMetric:
    """Tests for _evaluate_single_metric()."""

    @pytest.mark.asyncio
    async def test_success_observes_metric(
        self, mock_client, mocker: MockerFixture
    ) -> None:
        """On a valid judge response, observe() is called with the normalized score."""
        steps_response = _make_response_object("1. Step one\n2. Step two", mocker)
        score_response = _make_response_object('{"score": 4}', mocker)
        mock_client.responses.create.side_effect = [steps_response, score_response]

        mock_histogram = mocker.MagicMock()
        mock_labels = mocker.MagicMock()
        mock_histogram.labels.return_value = mock_labels
        mocker.patch("utils.llm_judge.metrics.llm_judge_score", mock_histogram)

        await _evaluate_single_metric(
            metric_name=METRIC_ANSWER_RELEVANCY,
            query="What is AI?",
            response="AI is artificial intelligence.",
            rag_context="",
            system_prompt="",
            model_id="provider/model",
            call_type="query",
            judge_model="judge/model",
            client=mock_client,
        )

        mock_histogram.labels.assert_called_once_with(
            provider="provider",
            model="model",
            call_type="query",
            metric_name=METRIC_ANSWER_RELEVANCY,
        )
        # score 4 normalizes to (4-1)/4 = 0.75
        mock_labels.observe.assert_called_once_with(pytest.approx(0.75))

    @pytest.mark.asyncio
    async def test_unparseable_response_does_not_observe(
        self, mock_client, mocker: MockerFixture
    ) -> None:
        """When the judge returns unparseable text, observe() is never called."""
        steps_response = _make_response_object("1. Step one", mocker)
        score_response = _make_response_object("I have no idea what to say.", mocker)
        mock_client.responses.create.side_effect = [steps_response, score_response]

        mock_histogram = mocker.MagicMock()
        mocker.patch("utils.llm_judge.metrics.llm_judge_score", mock_histogram)

        await _evaluate_single_metric(
            metric_name=METRIC_TOXICITY,
            query="q",
            response="r",
            rag_context="",
            system_prompt="",
            model_id="p/m",
            call_type="query",
            judge_model="j/m",
            client=mock_client,
        )

        mock_histogram.labels.return_value.observe.assert_not_called()

    @pytest.mark.asyncio
    async def test_step_generation_failure_falls_back_to_empty_steps(
        self, mock_client, mocker: MockerFixture
    ) -> None:
        """If step generation raises, scoring still runs with empty steps."""
        score_response = _make_response_object('{"score": 3}', mocker)
        mock_client.responses.create.side_effect = [
            RuntimeError("step gen failed"),
            score_response,
        ]

        mock_histogram = mocker.MagicMock()
        mock_labels = mocker.MagicMock()
        mock_histogram.labels.return_value = mock_labels
        mocker.patch("utils.llm_judge.metrics.llm_judge_score", mock_histogram)

        await _evaluate_single_metric(
            metric_name=METRIC_BIAS,
            query="q",
            response="r",
            rag_context="",
            system_prompt="",
            model_id="p/m",
            call_type="query",
            judge_model="j/m",
            client=mock_client,
        )

        # score 3 normalizes to 0.5
        mock_labels.observe.assert_called_once_with(pytest.approx(0.5))

    @pytest.mark.asyncio
    async def test_scoring_exception_does_not_propagate(
        self, mock_client, mocker: MockerFixture
    ) -> None:
        """If the scoring LLM call raises, the exception is swallowed."""
        steps_response = _make_response_object("1. Step one", mocker)
        mock_client.responses.create.side_effect = [
            steps_response,
            RuntimeError("scoring failed"),
        ]

        # Should not raise
        await _evaluate_single_metric(
            metric_name=METRIC_PROMPT_ALIGNMENT,
            query="q",
            response="r",
            rag_context="",
            system_prompt="",
            model_id="p/m",
            call_type="query",
            judge_model="j/m",
            client=mock_client,
        )


class TestEvaluateInteraction:
    """Tests for evaluate_interaction()."""

    @pytest.mark.asyncio
    async def test_all_seven_metrics_evaluated(
        self, mock_client, mocker: MockerFixture
    ) -> None:
        """evaluate_interaction calls _evaluate_single_metric for all 7 metrics."""
        called_metrics: list[str] = []

        async def fake_evaluate(metric_name: str, **_kwargs: object) -> None:
            called_metrics.append(metric_name)

        mocker.patch(
            "utils.llm_judge._evaluate_single_metric", side_effect=fake_evaluate
        )

        await evaluate_interaction(
            query="q",
            response="r",
            rag_chunks=[],
            system_prompt="",
            model_id="p/m",
            call_type="query",
            judge_model="j/m",
            client=mock_client,
        )

        assert sorted(called_metrics) == sorted(ALL_METRICS)

    @pytest.mark.asyncio
    async def test_rag_chunks_joined_correctly(
        self, mock_client, mocker: MockerFixture
    ) -> None:
        """RAG chunks are joined with double newlines into rag_context."""
        received_rag_contexts: list[str] = []

        async def capture_rag(
            metric_name: str, rag_context: str, **_kwargs: object
        ) -> None:
            received_rag_contexts.append(rag_context)

        mocker.patch("utils.llm_judge._evaluate_single_metric", side_effect=capture_rag)

        chunk1 = mocker.MagicMock()
        chunk1.content = "chunk one"
        chunk2 = mocker.MagicMock()
        chunk2.content = "chunk two"

        await evaluate_interaction(
            query="q",
            response="r",
            rag_chunks=[chunk1, chunk2],
            system_prompt="",
            model_id="p/m",
            call_type="query",
            judge_model="j/m",
            client=mock_client,
        )

        assert all(ctx == "chunk one\n\nchunk two" for ctx in received_rag_contexts)

    @pytest.mark.asyncio
    async def test_empty_rag_chunks_produces_empty_context(
        self, mock_client, mocker: MockerFixture
    ) -> None:
        """Empty rag_chunks list produces an empty rag_context string."""
        received_rag_contexts: list[str] = []

        async def capture_rag(
            metric_name: str, rag_context: str, **_kwargs: object
        ) -> None:
            received_rag_contexts.append(rag_context)

        mocker.patch("utils.llm_judge._evaluate_single_metric", side_effect=capture_rag)

        await evaluate_interaction(
            query="q",
            response="r",
            rag_chunks=[],
            system_prompt="",
            model_id="p/m",
            call_type="streaming_query",
            judge_model="j/m",
            client=mock_client,
        )

        assert all(ctx == "" for ctx in received_rag_contexts)

    @pytest.mark.asyncio
    async def test_individual_metric_failure_does_not_abort_others(
        self, mock_client, mocker: MockerFixture
    ) -> None:
        """If one metric raises, the remaining metrics still run."""
        call_count = 0

        async def flaky_evaluate(metric_name: str, **_kwargs: object) -> None:
            nonlocal call_count
            call_count += 1
            if metric_name == METRIC_TOXICITY:
                raise RuntimeError("judge error")

        mocker.patch(
            "utils.llm_judge._evaluate_single_metric", side_effect=flaky_evaluate
        )

        await evaluate_interaction(
            query="q",
            response="r",
            rag_chunks=[],
            system_prompt="",
            model_id="p/m",
            call_type="query",
            judge_model="j/m",
            client=mock_client,
        )

        assert call_count == len(ALL_METRICS)
