"""Unit tests for metrics/utils.py."""

import time

import pytest
from prometheus_client import Histogram
from pytest_mock import MockerFixture

from metrics.utils import measure_stream_duration, measure_ttft


async def _collect(ait) -> list:
    """Drain an async iterator and return all items as a list."""
    items = []
    async for item in ait:
        items.append(item)
    return items


async def _async_iter(items: list):
    """Yield items from a plain list as an async iterator."""
    for item in items:
        yield item


class TestMeasureStreamDuration:
    """Tests for measure_stream_duration."""

    @pytest.mark.asyncio
    async def test_yields_all_chunks(self, mocker: MockerFixture) -> None:
        """All chunks from the underlying stream are re-yielded unchanged."""
        mock_histogram = mocker.Mock(spec=Histogram)
        chunks = [1, 2, 3]

        result = await _collect(
            measure_stream_duration(
                time.monotonic(), mock_histogram, _async_iter(chunks)
            )
        )

        assert result == chunks

    @pytest.mark.asyncio
    async def test_histogram_observed_after_last_chunk(
        self, mocker: MockerFixture
    ) -> None:
        """Histogram is observed exactly once, after the stream is exhausted."""
        mock_histogram = mocker.Mock(spec=Histogram)

        await _collect(
            measure_stream_duration(
                time.monotonic(), mock_histogram, _async_iter([1, 2, 3])
            )
        )

        mock_histogram.observe.assert_called_once()
        elapsed = mock_histogram.observe.call_args[0][0]
        assert elapsed >= 0

    @pytest.mark.asyncio
    async def test_empty_stream_still_observes(self, mocker: MockerFixture) -> None:
        """Histogram is observed even when the stream yields no chunks."""
        mock_histogram = mocker.Mock(spec=Histogram)

        result = await _collect(
            measure_stream_duration(
                time.monotonic(), mock_histogram, _async_iter([])
            )
        )

        assert result == []
        mock_histogram.observe.assert_called_once()

    @pytest.mark.asyncio
    async def test_elapsed_time_is_non_negative(self, mocker: MockerFixture) -> None:
        """Observed elapsed time is non-negative."""
        mock_histogram = mocker.Mock(spec=Histogram)
        start = time.monotonic()

        await _collect(
            measure_stream_duration(start, mock_histogram, _async_iter(["a", "b"]))
        )

        elapsed = mock_histogram.observe.call_args[0][0]
        assert elapsed >= 0


class TestMeasureTTFT:
    """Tests for measure_ttft."""

    @pytest.mark.asyncio
    async def test_yields_all_chunks(self, mocker: MockerFixture) -> None:
        """All chunks from the underlying stream are re-yielded unchanged."""
        mock_histogram = mocker.Mock(spec=Histogram)
        chunks = ["a", "b", "c"]

        result = await _collect(
            measure_ttft(time.monotonic(), mock_histogram, _async_iter(chunks))
        )

        assert result == chunks

    @pytest.mark.asyncio
    async def test_histogram_observed_on_first_chunk_only(
        self, mocker: MockerFixture
    ) -> None:
        """Histogram is observed exactly once, on the first chunk."""
        mock_histogram = mocker.Mock(spec=Histogram)

        await _collect(
            measure_ttft(time.monotonic(), mock_histogram, _async_iter([1, 2, 3]))
        )

        mock_histogram.observe.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_stream_histogram_not_observed(
        self, mocker: MockerFixture
    ) -> None:
        """Histogram is never observed when the stream yields no chunks."""
        mock_histogram = mocker.Mock(spec=Histogram)

        result = await _collect(
            measure_ttft(time.monotonic(), mock_histogram, _async_iter([]))
        )

        assert result == []
        mock_histogram.observe.assert_not_called()

    @pytest.mark.asyncio
    async def test_elapsed_time_is_non_negative(self, mocker: MockerFixture) -> None:
        """Observed TTFT is non-negative."""
        mock_histogram = mocker.Mock(spec=Histogram)
        start = time.monotonic()

        await _collect(
            measure_ttft(start, mock_histogram, _async_iter(["first", "second"]))
        )

        elapsed = mock_histogram.observe.call_args[0][0]
        assert elapsed >= 0
