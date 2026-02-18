"""Unit tests for utils/query.py functions."""

import pytest
from fastapi import HTTPException

from app.endpoints.query import validate_attachments_metadata
from models.requests import Attachment, QueryRequest
from utils.query import prepare_input, prepare_text_input


def make_query_request(
    query: str = "test query",
    attachments: list[Attachment] | None = None,
) -> QueryRequest:
    """Build a minimal QueryRequest for testing.

    Args:
        query: The query string.
        attachments: Optional list of attachments.

    Returns:
        QueryRequest: A minimal query request instance.
    """
    return QueryRequest(query=query, attachments=attachments)


# ---------------------------------------------------------------------------
# prepare_text_input
# ---------------------------------------------------------------------------


def test_prepare_text_input_no_attachments() -> None:
    """prepare_text_input returns the bare query when there are no attachments."""
    request = make_query_request(query="hello")
    assert prepare_text_input(request) == "hello"


def test_prepare_text_input_text_attachment() -> None:
    """prepare_text_input appends text attachment content inline."""
    attachment = Attachment(
        attachment_type="log",
        content_type="text/plain",
        content="error: something went wrong",
    )
    request = make_query_request(query="what happened?", attachments=[attachment])
    result = prepare_text_input(request)
    assert result.startswith("what happened?")
    assert "[Attachment: log]" in result
    assert "error: something went wrong" in result


def test_prepare_text_input_image_attachment_uses_placeholder() -> None:
    """prepare_text_input replaces image content with <image> placeholder."""
    attachment = Attachment(
        attachment_type="screenshot",
        content_type="image/png",
        content="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk",
    )
    request = make_query_request(
        query="what is in this image?", attachments=[attachment]
    )
    result = prepare_text_input(request)
    assert "what is in this image?" in result
    assert "[Attachment: screenshot]" in result
    assert "<image>" in result
    # Raw base64 data must NOT appear in the text representation
    assert "iVBORw0KGgo" not in result


def test_prepare_text_input_mixed_attachments() -> None:
    """prepare_text_input handles both text and image attachments correctly."""
    text_attachment = Attachment(
        attachment_type="log",
        content_type="text/plain",
        content="log line",
    )
    image_attachment = Attachment(
        attachment_type="screenshot",
        content_type="image/jpeg",
        content="base64data",
    )
    request = make_query_request(
        query="analyse this",
        attachments=[text_attachment, image_attachment],
    )
    result = prepare_text_input(request)
    assert "log line" in result
    assert "<image>" in result
    assert "base64data" not in result


# ---------------------------------------------------------------------------
# prepare_input — text-only path
# ---------------------------------------------------------------------------


def test_prepare_input_no_attachments_returns_string() -> None:
    """prepare_input returns a plain string when there are no attachments."""
    request = make_query_request(query="hello")
    result = prepare_input(request)
    assert isinstance(result, str)
    assert result == "hello"


def test_prepare_input_text_attachment_returns_string() -> None:
    """prepare_input returns a plain string for text-only attachments."""
    attachment = Attachment(
        attachment_type="configuration",
        content_type="application/yaml",
        content="key: value",
    )
    request = make_query_request(query="explain this config", attachments=[attachment])
    result = prepare_input(request)
    assert isinstance(result, str)
    assert "explain this config" in result
    assert "[Attachment: configuration]" in result
    assert "key: value" in result


# ---------------------------------------------------------------------------
# prepare_input — multimodal path
# ---------------------------------------------------------------------------


def test_prepare_input_image_attachment_returns_list() -> None:
    """prepare_input returns a message list when an image attachment is present."""
    attachment = Attachment(
        attachment_type="screenshot",
        content_type="image/png",
        content="abc123",
    )
    request = make_query_request(query="describe this", attachments=[attachment])
    result = prepare_input(request)
    assert isinstance(result, list)
    assert len(result) == 1
    message = result[0]
    assert message["role"] == "user"
    assert message["type"] == "message"
    content = message["content"]
    assert isinstance(content, list)


def test_prepare_input_image_attachment_content_items() -> None:
    """prepare_input multimodal list contains correct text and image items."""
    attachment = Attachment(
        attachment_type="screenshot",
        content_type="image/png",
        content="abc123",
    )
    request = make_query_request(query="describe this", attachments=[attachment])
    result = prepare_input(request)
    assert isinstance(result, list)
    content = result[0]["content"]

    text_items = [item for item in content if item["type"] == "input_text"]
    image_items = [item for item in content if item["type"] == "input_image"]

    assert len(text_items) == 1
    assert "describe this" in text_items[0]["text"]

    assert len(image_items) == 1
    assert image_items[0]["detail"] == "auto"
    assert image_items[0]["image_url"] == "data:image/png;base64,abc123"


def test_prepare_input_multiple_image_attachments() -> None:
    """prepare_input produces one image item per image attachment."""
    attachments = [
        Attachment(
            attachment_type="screenshot", content_type="image/png", content="img1"
        ),
        Attachment(
            attachment_type="screenshot", content_type="image/jpeg", content="img2"
        ),
    ]
    request = make_query_request(query="compare", attachments=attachments)
    result = prepare_input(request)
    assert isinstance(result, list)
    content = result[0]["content"]
    image_items = [item for item in content if item["type"] == "input_image"]
    assert len(image_items) == 2
    assert image_items[0]["image_url"] == "data:image/png;base64,img1"
    assert image_items[1]["image_url"] == "data:image/jpeg;base64,img2"


def test_prepare_input_mixed_attachments_multimodal() -> None:
    """prepare_input includes text attachment content in the text item, not as separate image."""
    text_attachment = Attachment(
        attachment_type="log",
        content_type="text/plain",
        content="log line",
    )
    image_attachment = Attachment(
        attachment_type="screenshot",
        content_type="image/webp",
        content="imgdata",
    )
    request = make_query_request(
        query="analyse",
        attachments=[text_attachment, image_attachment],
    )
    result = prepare_input(request)
    assert isinstance(result, list)
    content = result[0]["content"]

    text_items = [item for item in content if item["type"] == "input_text"]
    image_items = [item for item in content if item["type"] == "input_image"]

    assert len(text_items) == 1
    assert "log line" in text_items[0]["text"]
    assert len(image_items) == 1
    assert "imgdata" in image_items[0]["image_url"]


def test_prepare_input_all_supported_image_content_types() -> None:
    """prepare_input treats all supported image MIME types as image attachments."""
    for mime in ("image/png", "image/jpeg", "image/gif", "image/webp"):
        attachment = Attachment(
            attachment_type="screenshot",
            content_type=mime,
            content="data",
        )
        request = make_query_request(query="q", attachments=[attachment])
        result = prepare_input(request)
        assert isinstance(result, list), f"Expected list for {mime}"
        content = result[0]["content"]
        image_items = [item for item in content if item["type"] == "input_image"]
        assert len(image_items) == 1
        assert image_items[0]["image_url"] == f"data:{mime};base64,data"


# ---------------------------------------------------------------------------
# validate_attachments_metadata — screenshot / image types
# ---------------------------------------------------------------------------


def test_validate_attachments_metadata_screenshot_type() -> None:
    """validate_attachments_metadata accepts the 'screenshot' attachment type."""
    attachments = [
        Attachment(
            attachment_type="screenshot",
            content_type="image/png",
            content="abc123",
        )
    ]
    # Should not raise
    validate_attachments_metadata(attachments)


@pytest.mark.parametrize("mime", ["image/png", "image/jpeg", "image/gif", "image/webp"])
def test_validate_attachments_metadata_image_content_types(mime: str) -> None:
    """validate_attachments_metadata accepts all supported image MIME types."""
    attachments = [
        Attachment(attachment_type="screenshot", content_type=mime, content="data")
    ]
    validate_attachments_metadata(attachments)


def test_validate_attachments_metadata_invalid_image_content_type() -> None:
    """validate_attachments_metadata rejects unsupported image MIME types."""
    attachments = [
        Attachment(
            attachment_type="screenshot",
            content_type="image/bmp",
            content="data",
        )
    ]
    with pytest.raises(HTTPException) as exc_info:
        validate_attachments_metadata(attachments)
    assert exc_info.value.status_code == 422
