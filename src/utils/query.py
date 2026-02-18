"""Utility functions for working with queries."""

import json
from typing import Any, AsyncIterator, Optional

import constants
from models.requests import QueryRequest


from llama_stack.apis.agents.openai_responses import (
    OpenAIResponseContentPartOutputText,
    OpenAIResponseObject,
    OpenAIResponseObjectStream,
    OpenAIResponseObjectStreamResponseCreated,
    OpenAIResponseObjectStreamResponseContentPartAdded,
    OpenAIResponseObjectStreamResponseOutputTextDelta,
    OpenAIResponseObjectStreamResponseOutputTextDone,
    OpenAIResponseMessage,
    OpenAIResponseOutputMessageContentOutputText,
    OpenAIResponseObjectStreamResponseCompleted,
)


def prepare_text_input(query_request: QueryRequest) -> str:
    """Prepare a plain-text representation of the query and its attachments.

    Always returns a string regardless of attachment types. Image attachments are
    represented as ``<image>`` placeholders so that the string remains safe for
    text-only consumers such as shield moderation or conversation logging.

    Args:
        query_request: The query request containing the query and optional attachments.

    Returns:
        str: The query text with attachment labels appended. Image attachments are
            replaced by a ``<image>`` placeholder instead of their base64 content.
    """
    input_text = query_request.query
    for attachment in query_request.attachments or []:
        if attachment.content_type in constants.ATTACHMENT_IMAGE_CONTENT_TYPES:
            input_text += f"\n\n[Attachment: {attachment.attachment_type}] <image>"
        else:
            input_text += (
                f"\n\n[Attachment: {attachment.attachment_type}]\n{attachment.content}"
            )
    return input_text


def prepare_input(query_request: QueryRequest) -> str | list[Any]:
    """Prepare input for the Responses API, supporting both text and image attachments.

    Returns a plain string when there are no image attachments (backward-compatible
    behaviour). Returns a multimodal message list when at least one image attachment
    is present so that the model can process the image alongside the text.

    The multimodal list follows the OpenAI Responses API format::

        [
            {
                "role": "user",
                "type": "message",
                "content": [
                    {"type": "input_text", "text": "<query + text attachments>"},
                    {"type": "input_image", "detail": "auto",
                     "image_url": "data:image/png;base64,..."},
                ],
            }
        ]

    Args:
        query_request: The query request containing the query and optional attachments.

    Returns:
        str | list[Any]: A plain string for text-only requests, or a message list for
            requests that include one or more image attachments.
    """
    attachments = query_request.attachments or []
    has_images = any(
        a.content_type in constants.ATTACHMENT_IMAGE_CONTENT_TYPES for a in attachments
    )

    if not has_images:
        input_text = query_request.query
        for attachment in attachments:
            input_text += (
                f"\n\n[Attachment: {attachment.attachment_type}]\n{attachment.content}"
            )
        return input_text

    # Multimodal: build text content from query + non-image attachments
    text_content = query_request.query
    for attachment in attachments:
        if attachment.content_type not in constants.ATTACHMENT_IMAGE_CONTENT_TYPES:
            text_content += (
                f"\n\n[Attachment: {attachment.attachment_type}]\n{attachment.content}"
            )

    content_items: list[dict[str, str]] = [{"type": "input_text", "text": text_content}]
    for attachment in attachments:
        if attachment.content_type in constants.ATTACHMENT_IMAGE_CONTENT_TYPES:
            content_items.append(
                {
                    "type": "input_image",
                    "detail": "auto",
                    "image_url": (
                        f"data:{attachment.content_type};base64,{attachment.content}"
                    ),
                }
            )

    return [{"role": "user", "type": "message", "content": content_items}]


def parse_arguments_string(arguments_str: str) -> dict[str, Any]:
    """
    Try to parse an arguments string into a dictionary.

    Attempts multiple parsing strategies:
    1. Try parsing the string as-is as JSON (if it's already valid JSON)
    2. Try wrapping the string in {} if it doesn't start with {
    3. Return {"args": arguments_str} if all attempts fail

    Args:
        arguments_str: The arguments string to parse

    Returns:
        Parsed dictionary if successful, otherwise {"args": arguments_str}
    """
    # Try parsing as-is first (most common case)
    try:
        parsed = json.loads(arguments_str)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass

    # Try wrapping in {} if string doesn't start with {
    # This handles cases where the string is just the content without braces
    stripped = arguments_str.strip()
    if stripped and not stripped.startswith("{"):
        try:
            wrapped = "{" + stripped + "}"
            parsed = json.loads(wrapped)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass

    # Fallback: return wrapped in arguments key
    return {"args": arguments_str}


async def create_violation_stream(
    message: str,
    shield_model: Optional[str] = None,
) -> AsyncIterator[OpenAIResponseObjectStream]:
    """Generate a minimal streaming response for cases where input is blocked by a shield.

    This yields only the essential streaming events to indicate that the input was rejected.
    Dummy item identifiers are used solely for protocol compliance and are not used later.
    """
    response_id = "resp_shield_violation"

    # Create the response object with empty output at the beginning
    response_obj = OpenAIResponseObject(
        id=response_id,
        created_at=0,  # not used
        model=shield_model or "shield",
        output=[],
        status="in_progress",
    )
    yield OpenAIResponseObjectStreamResponseCreated(response=response_obj)

    # Triggers empty initial token
    yield OpenAIResponseObjectStreamResponseContentPartAdded(
        content_index=0,
        response_id=response_id,
        item_id="msg_shield_violation_1",
        output_index=0,
        part=OpenAIResponseContentPartOutputText(text=""),
        sequence_number=0,
    )

    # Text delta
    yield OpenAIResponseObjectStreamResponseOutputTextDelta(
        content_index=1,
        delta=message,
        item_id="msg_shield_violation_2",
        output_index=1,
        sequence_number=1,
    )

    # Output text done
    yield OpenAIResponseObjectStreamResponseOutputTextDone(
        content_index=2,
        text=message,
        item_id="msg_shield_violation_3",
        output_index=2,
        sequence_number=2,
    )

    # Fill the output when message is completed
    response_obj.output = [
        OpenAIResponseMessage(
            id="msg_shield_violation_4",
            content=[OpenAIResponseOutputMessageContentOutputText(text=message)],
            role="assistant",
            status="completed",
        )
    ]
    # Update status to completed
    response_obj.status = "completed"

    # Completed response triggers turn complete event
    yield OpenAIResponseObjectStreamResponseCompleted(response=response_obj)
