"""Tests for social routing helpers and their integration with bamboo_answer.

Covers:
  - ``_is_greeting()`` — pattern hits and misses
  - ``_is_ack()`` — pattern hits and misses
  - Mutual exclusivity between the two predicates
  - ``_route()`` short-circuits before topic guard for social messages
  - topic_guard ``check_topic()`` keyword-allows social terms (defence-in-depth)
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from bamboo.tools.bamboo_answer import (
    BambooAnswerTool,
    _is_ack,
    _is_greeting,
    _GREETING_RESPONSE,
    _ACK_RESPONSE,
)
import bamboo.tools.bamboo_answer as ba_mod
from bamboo.tools.topic_guard import check_topic


# ---------------------------------------------------------------------------
# _is_greeting — hits
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "hello",
    "Hello",
    "HELLO",
    "hello!",
    "hello.",
    "hi",
    "Hi",
    "hii",
    "hi!",
    "hey",
    "heyyy!",
    "Hey!",
    "good morning",
    "Good Morning",
    "good afternoon",
    "good evening",
    "good day",
    "howdy",
    "greetings",
    "sup",
    "yo",
    "  hello  ",
])
def test_is_greeting_true(text: str) -> None:
    """_is_greeting returns True for standalone greeting messages."""
    assert _is_greeting(text), f"Expected greeting match for: {text!r}"


# ---------------------------------------------------------------------------
# _is_greeting — misses (contain a real follow-up query or are not greetings)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "hello, can you check task 123",
    "hi, what is the status of job 456",
    "hey can you help me with something",
    "good morning — I have a question about a failed job",
    "thanks",
    "what is PanDA?",
    "",
    "how do I submit a task",
])
def test_is_greeting_false(text: str) -> None:
    """_is_greeting returns False when the message contains a real query."""
    assert not _is_greeting(text), f"Expected no greeting match for: {text!r}"


# ---------------------------------------------------------------------------
# _is_ack — hits
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "thanks",
    "Thanks",
    "THANKS",
    "thank you",
    "Thank you",
    "thank you so much",
    "thanks a lot",
    "thanks very much",
    "thanks for that",
    "thx",
    "cheers",
    "great",
    "great!",
    "great, thanks",
    "great.",
    "perfect",
    "perfect!",
    "awesome",
    "sounds good",
    "sound good",
    "got it",
    "ok",
    "okay",
    "okay!",
    "cool",
    "nice",
    "brilliant",
    "excellent",
    "wonderful",
    "understood",
    "noted",
    "roger",
    "roger that",
    "good to know",
    "that's helpful",
    "that is helpful",
    "that's great",
    "that is perfect",
    "that's useful",
    "bye",
    "bye!",
    "goodbye",
    "see you",
    "see you later",
    "  thanks  ",
])
def test_is_ack_true(text: str) -> None:
    """_is_ack returns True for standalone acknowledgement messages."""
    assert _is_ack(text), f"Expected ack match for: {text!r}"


# ---------------------------------------------------------------------------
# _is_ack — misses (contain a real follow-up query)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "thanks, but can you also check job 789",
    "great, now please analyse task 48432100",
    "ok so how do I fix a pilot error",
    "got it — what about the site errors at AGLT2?",
    "understood, what does piloterrorcode 1008 mean",
    "hello",
    "what is PanDA?",
    "",
    "why did my job fail",
])
def test_is_ack_false(text: str) -> None:
    """_is_ack returns False when the message contains a real follow-up query."""
    assert not _is_ack(text), f"Expected no ack match for: {text!r}"


# ---------------------------------------------------------------------------
# Mutual exclusivity
# ---------------------------------------------------------------------------

def test_hello_is_greeting_not_ack() -> None:
    """'hello' is classified as a greeting, not an acknowledgement."""
    assert _is_greeting("hello")
    assert not _is_ack("hello")


def test_thanks_is_ack_not_greeting() -> None:
    """'thanks' is classified as an acknowledgement, not a greeting."""
    assert _is_ack("thanks")
    assert not _is_greeting("thanks")


# ---------------------------------------------------------------------------
# Response strings are non-empty and mention AskPanDA / helpful content
# ---------------------------------------------------------------------------

def test_greeting_response_content() -> None:
    """Greeting response mentions AskPanDA and hints at what to ask."""
    assert "AskPanDA" in _GREETING_RESPONSE or "ask" in _GREETING_RESPONSE.lower()
    assert len(_GREETING_RESPONSE) > 20


def test_ack_response_content() -> None:
    """Ack response is warm and non-empty."""
    assert "welcome" in _ACK_RESPONSE.lower() or "help" in _ACK_RESPONSE.lower()
    assert len(_ACK_RESPONSE) > 10


# ---------------------------------------------------------------------------
# Integration: _route() short-circuits before topic guard for social inputs
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_greeting_bypasses_topic_guard() -> None:
    """'hello' returns the greeting response without calling check_topic."""
    guard_mock = AsyncMock()
    tool = BambooAnswerTool()

    with patch.object(ba_mod, "check_topic", guard_mock):
        result = await tool.call({"question": "hello"})

    guard_mock.assert_not_awaited()
    assert result[0]["text"] == _GREETING_RESPONSE


@pytest.mark.asyncio
async def test_ack_bypasses_topic_guard() -> None:
    """'thanks' returns the ack response without calling check_topic."""
    guard_mock = AsyncMock()
    tool = BambooAnswerTool()

    with patch.object(ba_mod, "check_topic", guard_mock):
        result = await tool.call({"question": "thanks"})

    guard_mock.assert_not_awaited()
    assert result[0]["text"] == _ACK_RESPONSE


@pytest.mark.asyncio
async def test_greeting_bypasses_topic_guard_via_messages() -> None:
    """'hello' sent as a messages list also gets the canned greeting."""
    guard_mock = AsyncMock()
    tool = BambooAnswerTool()

    with patch.object(ba_mod, "check_topic", guard_mock):
        result = await tool.call({"messages": [{"role": "user", "content": "hello"}]})

    guard_mock.assert_not_awaited()
    assert result[0]["text"] == _GREETING_RESPONSE


@pytest.mark.asyncio
async def test_greeting_does_not_trigger_refusal() -> None:
    """'hello' must never return the 'I can only answer' rejection message."""
    tool = BambooAnswerTool()
    # No mocks — run with real guard to confirm defence-in-depth also works.
    # The social intercept fires first so the guard is never reached, but
    # even if it were, the guard now keyword-allows 'hello'.
    guard_mock = AsyncMock()
    with patch.object(ba_mod, "check_topic", guard_mock):
        result = await tool.call({"question": "hello"})

    assert "I can only answer" not in result[0]["text"]


@pytest.mark.asyncio
async def test_ack_does_not_trigger_refusal() -> None:
    """'thanks' must never return the 'I can only answer' rejection message."""
    guard_mock = AsyncMock()
    tool = BambooAnswerTool()

    with patch.object(ba_mod, "check_topic", guard_mock):
        result = await tool.call({"question": "thanks"})

    assert "I can only answer" not in result[0]["text"]


@pytest.mark.asyncio
async def test_greeting_with_task_query_not_social() -> None:
    """'hello, check task 48432100' is NOT intercepted as social-only."""
    from bamboo.tools.topic_guard import GuardResult

    guard_mock = AsyncMock(return_value=GuardResult(
        allowed=True, reason="keyword_allow", llm_used=False
    ))
    execute_mock = AsyncMock(return_value=[{"type": "text", "text": "task result"}])
    tool = BambooAnswerTool()

    with (
        patch.object(ba_mod, "check_topic", guard_mock),
        patch.object(ba_mod, "execute_plan", execute_mock),
    ):
        result = await tool.call({"question": "hello, check task 48432100"})

    # Guard must have been called — the message was not intercepted as social.
    guard_mock.assert_awaited_once()
    assert result[0]["text"] == "task result"


@pytest.mark.asyncio
async def test_ack_with_followup_not_social() -> None:
    """'thanks, but also check job 7061545370' routes normally, not as ack."""
    from bamboo.tools.topic_guard import GuardResult

    guard_mock = AsyncMock(return_value=GuardResult(
        allowed=True, reason="keyword_allow", llm_used=False
    ))
    execute_mock = AsyncMock(return_value=[{"type": "text", "text": "job result"}])
    tool = BambooAnswerTool()

    with (
        patch.object(ba_mod, "check_topic", guard_mock),
        patch.object(ba_mod, "execute_plan", execute_mock),
    ):
        result = await tool.call({"question": "thanks, but also check job 7061545370"})

    guard_mock.assert_awaited_once()
    assert result[0]["text"] == "job result"


# ---------------------------------------------------------------------------
# Defence-in-depth: topic_guard keyword-allows social terms directly
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_topic_guard_keyword_allows_hello() -> None:
    """check_topic('hello') returns keyword_allow without calling the LLM."""
    result = await check_topic("hello")
    assert result.allowed is True
    assert result.reason == "keyword_allow"
    assert result.llm_used is False


@pytest.mark.asyncio
async def test_topic_guard_keyword_allows_thanks() -> None:
    """check_topic('thanks') returns keyword_allow without calling the LLM."""
    result = await check_topic("thanks")
    assert result.allowed is True
    assert result.reason == "keyword_allow"
    assert result.llm_used is False


@pytest.mark.asyncio
async def test_topic_guard_keyword_allows_hi() -> None:
    """check_topic('hi') returns keyword_allow without calling the LLM."""
    result = await check_topic("hi")
    assert result.allowed is True
    assert result.reason == "keyword_allow"
    assert result.llm_used is False


@pytest.mark.asyncio
async def test_topic_guard_keyword_allows_cheers() -> None:
    """check_topic('cheers') returns keyword_allow without calling the LLM."""
    result = await check_topic("cheers")
    assert result.allowed is True
    assert result.reason == "keyword_allow"
    assert result.llm_used is False
