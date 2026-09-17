"""Tests for Telegram human-decision prompts."""

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter
from tools import clarify_gateway as cm

def _clear_state():
    with cm._lock:
        cm._entries.clear()
        cm._session_index.clear()
        cm._notify_cbs.clear()

def _make_adapter():
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token", extra={}))
    adapter._bot = AsyncMock()
    adapter._app = MagicMock()
    return adapter

def _query(data, user_id="777"):
    query = AsyncMock()
    query.data = data
    query.message = MagicMock()
    query.message.chat_id = 12345
    query.message.text = "Question"
    query.message.message_thread_id = None
    query.message.chat.type = "private"
    query.from_user = SimpleNamespace(id=user_id, first_name="Tester")
    return query

class TestHumanDecisionPrimitive:
    def setup_method(self):
        _clear_state()

    def test_first_writer_wins_and_timeout_zero_is_indefinite(self):
        cm.register_human_decision("decision-1", "session-1", "Proceed?")
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(
                lambda answer: cm.resolve_human_decision("decision-1", answer),
                ("yes", "no"),
            ))
        assert sorted(results) == [False, True]
        assert cm.wait_for_human_decision("decision-1", timeout=0) in {"yes", "no"}

    def test_custom_enables_verbatim_text_resolution(self):
        entry = cm.register_human_decision("decision-2", "session-2", "Proceed?")
        assert entry.awaiting_text is False
        assert cm.mark_human_decision_awaiting_text("decision-2") is True
        assert entry.awaiting_text is True
        response = "  preserve this exactly  \n"
        assert cm.attempt_text_response_for_session("session-2", response) == cm.TEXT_RESOLVED
        assert cm.wait_for_human_decision("decision-2", timeout=1) == response

class TestTelegramHumanDecision:
    def setup_method(self):
        _clear_state()

    def test_sends_exact_three_buttons(self, monkeypatch):
        adapter = _make_adapter()
        adapter._bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=42))
        buttons = []
        monkeypatch.setattr(
            "plugins.platforms.telegram.adapter.InlineKeyboardButton",
            lambda text, callback_data: buttons.append((text, callback_data)) or (text, callback_data),
        )
        monkeypatch.setattr(
            "plugins.platforms.telegram.adapter.InlineKeyboardMarkup",
            lambda rows: rows,
        )

        result = asyncio.run(adapter.send_human_decision(
            chat_id="12345", question="?", decision_id="d1", session_key="s1",
        ))

        assert result.success is True
        assert [label for label, _ in buttons] == ["Yes", "No", "Custom text"]
        assert [data for _, data in buttons] == ["hd:d1:yes", "hd:d1:no", "hd:d1:custom"]
        assert adapter._human_decision_state["d1"] == "s1"

    def test_callback_requires_auth_and_yes_edits_keyboard_away(self):
        adapter = _make_adapter()
        cm.register_human_decision("d2", "s2", "Proceed?")
        adapter._human_decision_state["d2"] = "s2"
        query = _query("hd:d2:yes", user_id="999")
        update = SimpleNamespace(callback_query=query)

        with patch.object(adapter, "_callback_authorized", new=AsyncMock(return_value=False)) as authorized:
            asyncio.run(adapter._handle_callback_query(update, MagicMock()))
        authorized.assert_awaited_once()
        with cm._lock:
            assert not cm._entries["d2"].event.is_set()

        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            asyncio.run(adapter._handle_callback_query(update, MagicMock()))
        with cm._lock:
            assert cm._entries["d2"].response == "yes"
        query.edit_message_text.assert_awaited_once()
        assert query.edit_message_text.call_args.kwargs["reply_markup"] is None

    def test_custom_callback_then_inbound_text_resolves_verbatim(self):
        adapter = _make_adapter()
        cm.register_human_decision("d3", "s3", "Proceed?")
        adapter._human_decision_state["d3"] = "s3"
        query = _query("hd:d3:custom")
        update = SimpleNamespace(callback_query=query)

        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            asyncio.run(adapter._handle_callback_query(update, MagicMock()))

        with cm._lock:
            assert cm._entries["d3"].awaiting_text is True
        response = " exact custom answer "
        assert cm.attempt_text_response_for_session("s3", response) == cm.TEXT_RESOLVED
        assert cm.wait_for_human_decision("d3", timeout=1) == response
        assert query.edit_message_text.call_args.kwargs["reply_markup"] is None

    def test_callback_rejects_wrong_chat_thread_and_session(self):
        adapter = _make_adapter()
        cm.register_human_decision("d4", "s4", "Proceed?")
        adapter._human_decision_state["d4"] = "s4"
        query = _query("hd:d4:yes", user_id="999")
        update = SimpleNamespace(callback_query=query)

        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            asyncio.run(adapter._handle_callback_query(update, MagicMock()))

        with cm._lock:
            assert not cm._entries["d4"].event.is_set()
        query.answer.assert_awaited_once()
        assert query.answer.call_args[1]["text"] == "⛔ You are not authorized to answer this prompt."
        assert adapter._human_decision_state["d4"] == "s4"

    def test_clear_session_clears_adapter_callback_state(self):
        adapter = _make_adapter()
        cm.register_human_decision("d5", "s5", "Proceed?")
        adapter._human_decision_state["d5"] = "s5"
        cm.clear_session("s5")
        with cm._lock:
            assert "d5" not in cm._entries
        assert "d5" not in adapter._human_decision_state
