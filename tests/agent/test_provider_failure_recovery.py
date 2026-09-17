"""Tests for provider failure recovery — deterministic transient-provider-failure
recovery for Telegram-facing agents.

Covers:
- admission-busy error classification (429/503 with admission/busy patterns)
- 5-attempt maximum recovery within 120 seconds wall-clock
- bounded exponential backoff honoring Retry-After
- non-retryable classification (auth, policy, quota exhaustion)
- context compaction reducing request size before retry
- duplicate side-effect tool execution prevention
- duplicate final delivery prevention (idempotency)
- Telegram-safe terminal message on exhaustion
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ── Helpers ──────────────────────────────────────────────────────────────

class _MockAPIError(Exception):
    """Simulates an OpenAI SDK APIStatusError."""
    def __init__(self, message, *, status_code=None, body=None, headers=None, response=None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body or {}
        self.headers = headers or {}
        self.response = response or SimpleNamespace(headers=headers or {})


class _MockAgent:
    """Minimal agent stub for provider-failure-recovery tests."""

    def __init__(self, **overrides):
        self.provider = overrides.get("provider", "test-provider")
        self.model = overrides.get("model", "test-model")
        self.base_url = overrides.get("base_url", "https://api.test.com/v1")
        self._buffered_status = []
        self._emitted_status = []
        self.log_prefix = "[test] "
        self.platform = overrides.get("platform", "telegram")

    def _buffer_status(self, msg):
        self._buffered_status.append(msg)

    def _emit_status(self, msg):
        self._emitted_status.append(msg)

    def _flush_status_buffer(self):
        self._buffered_status.clear()

    def _touch_activity(self, label):
        pass

    def _summarize_api_error(self, error):
        return str(error)[:200]


# ── Test: FailoverReason.admission_busy exists ──────────────────────────

class TestAdmissionBusyFailoverReason:
    """The admission_busy variant must be present in the enum and behave like a
    transient retryable failure."""

    def test_admission_busy_in_enum(self):
        from agent.error_classifier import FailoverReason
        assert hasattr(FailoverReason, "admission_busy")
        assert FailoverReason.admission_busy.value == "admission_busy"

    def test_admission_busy_is_retryable(self):
        from agent.error_classifier import ClassifiedError, FailoverReason
        ce = ClassifiedError(reason=FailoverReason.admission_busy, retryable=True)
        assert ce.retryable is True

    def test_admission_busy_not_auth(self):
        from agent.error_classifier import ClassifiedError, FailoverReason
        ce = ClassifiedError(reason=FailoverReason.admission_busy)
        assert not ce.is_auth


# ── Test: Admission-busy error classification ───────────────────────────

class TestClassifyAdmissionBusy:
    """Admission-busy errors must be classified correctly from various shapes."""

    def _classify(self, message, *, status_code=429, body=None, provider="nous"):
        from agent.error_classifier import classify_api_error
        error = _MockAPIError(message, status_code=status_code, body=body or {})
        return classify_api_error(error, provider=provider, model="test-model")

    def test_chat_admission_busy_429(self):
        result = self._classify("chat_admission_busy: provider is busy", status_code=429)
        assert result.reason.value == "admission_busy"
        assert result.retryable is True

    def test_admission_busy_503(self):
        result = self._classify("admission-busy: service temporarily unavailable", status_code=503)
        assert result.reason.value == "admission_busy"
        assert result.retryable is True

    def test_admission_busy_generic_429(self):
        result = self._classify("admission busy", status_code=429)
        assert result.reason.value == "admission_busy"
        assert result.retryable is True

    def test_admission_busy_should_not_fallback_immediately(self):
        """admission_busy is transient — fallback is not needed on the first attempt."""
        result = self._classify("chat_admission_busy", status_code=429)
        # Retryable, not a billing/auth/format error
        assert result.retryable is True
        assert result.should_fallback is False


# ── Test: Non-retryable classification ──────────────────────────────────

class TestNonRetryableClassification:
    """Certain error classes must NEVER be retried."""

    def test_auth_permanent_not_retried(self):
        from agent.error_classifier import classify_api_error, FailoverReason
        error = _MockAPIError("Invalid API key", status_code=401)
        result = classify_api_error(error, provider="openai")
        assert result.retryable is False
        assert result.reason in {FailoverReason.auth, FailoverReason.auth_permanent}

    def test_billing_exhaustion_not_retried(self):
        from agent.error_classifier import classify_api_error, FailoverReason
        error = _MockAPIError("quota exceeded", status_code=402)
        result = classify_api_error(error, provider="openai")
        assert result.reason == FailoverReason.billing
        assert result.retryable is False

    def test_content_policy_not_retried(self):
        from agent.error_classifier import classify_api_error, FailoverReason
        error = _MockAPIError("violates our usage policies", status_code=400)
        result = classify_api_error(error, provider="openai")
        assert result.reason == FailoverReason.content_policy_blocked
        assert result.retryable is False

    def test_malformed_request_not_retried(self):
        from agent.error_classifier import classify_api_error, FailoverReason
        error = _MockAPIError("unknown parameter: foo", status_code=400)
        result = classify_api_error(error, provider="openai")
        assert result.reason == FailoverReason.format_error
        assert result.retryable is False


# ── Test: Provider failure recovery limits ──────────────────────────────

class TestProviderFailureRecoveryLimits:
    """5 total recovery attempts for one original request, bounded within 120 seconds."""

    def test_max_attempts_constant(self):
        from agent.provider_failure_recovery import (
            PROVIDER_RECOVERY_MAX_ATTEMPTS,
            PROVIDER_RECOVERY_WINDOW_SECONDS,
        )
        assert PROVIDER_RECOVERY_MAX_ATTEMPTS == 5
        assert PROVIDER_RECOVERY_WINDOW_SECONDS == 120.0

    def test_recovery_state_tracks_attempts_and_start_time(self):
        from agent.provider_failure_recovery import ProviderRecoveryState
        state = ProviderRecoveryState()
        assert state.attempts == 0
        assert state.start_time is None
        assert state.error_class is None

    def test_recovery_state_increment(self):
        from agent.provider_failure_recovery import ProviderRecoveryState
        state = ProviderRecoveryState()
        now = time.monotonic()
        state.start(now, "admission_busy")
        state.increment()
        assert state.attempts == 1
        state.increment()
        assert state.attempts == 2

    def test_recovery_state_exhausted_at_5(self):
        from agent.provider_failure_recovery import (
            ProviderRecoveryState,
            PROVIDER_RECOVERY_MAX_ATTEMPTS,
        )
        state = ProviderRecoveryState()
        state.start(time.monotonic(), "admission_busy")
        for _ in range(PROVIDER_RECOVERY_MAX_ATTEMPTS):
            assert not state.is_exhausted()
            state.increment()
        assert state.is_exhausted()

    def test_recovery_state_within_window(self):
        from agent.provider_failure_recovery import (
            ProviderRecoveryState,
            PROVIDER_RECOVERY_WINDOW_SECONDS,
        )
        state = ProviderRecoveryState()
        now = time.monotonic()
        state.start(now, "admission_busy")
        # Within window
        assert not state.window_expired(now + PROVIDER_RECOVERY_WINDOW_SECONDS - 1)
        # At/past window
        assert state.window_expired(now + PROVIDER_RECOVERY_WINDOW_SECONDS)
        assert state.window_expired(now + PROVIDER_RECOVERY_WINDOW_SECONDS + 10)

    def test_recovery_state_same_error_class(self):
        from agent.provider_failure_recovery import ProviderRecoveryState
        state = ProviderRecoveryState()
        state.start(time.monotonic(), "admission_busy")
        assert state.is_same_error_class("admission_busy")
        assert not state.is_same_error_class("timeout")

    def test_recovery_state_reset(self):
        from agent.provider_failure_recovery import ProviderRecoveryState
        state = ProviderRecoveryState()
        state.start(time.monotonic(), "admission_busy")
        state.increment()
        state.increment()
        state.reset()
        assert state.attempts == 0
        assert state.start_time is None


# ── Test: Bounded exponential backoff ───────────────────────────────────

class TestBoundedBackoff:
    """Backoff must be exponential, bounded, and respect Retry-After."""

    def test_backoff_increases_monotonically(self):
        from agent.provider_failure_recovery import compute_recovery_backoff
        delays = []
        for attempt in range(5):
            d = compute_recovery_backoff(attempt, retry_after=None)
            delays.append(d)
        # Each delay >= previous (within jitter tolerance)
        for i in range(1, len(delays)):
            assert delays[i] >= delays[i - 1] * 0.5, (
                f"delay[{i}]={delays[i]} < delay[{i-1}]={delays[i-1]} * 0.5"
            )

    def test_backoff_bounded_by_max(self):
        from agent.provider_failure_recovery import compute_recovery_backoff, _PROVIDER_RECOVERY_MAX_BACKOFF
        for attempt in range(10):
            d = compute_recovery_backoff(attempt, retry_after=None)
            # Jitter can push slightly above the max, so allow 1.5x margin
            assert d <= _PROVIDER_RECOVERY_MAX_BACKOFF * 1.5, (
                f"delay={d} exceeds max={_PROVIDER_RECOVERY_MAX_BACKOFF}"
            )

    def test_retry_after_honored(self):
        from agent.provider_failure_recovery import compute_recovery_backoff
        d = compute_recovery_backoff(0, retry_after=30.0)
        assert d >= 30.0

    def test_retry_after_capped(self):
        from agent.provider_failure_recovery import compute_recovery_backoff
        d = compute_recovery_backoff(0, retry_after=600.0)
        assert d <= 120.0

    def test_retry_after_zero_ignored(self):
        from agent.provider_failure_recovery import compute_recovery_backoff
        d = compute_recovery_backoff(0, retry_after=0.0)
        assert d > 0.0  # Falls back to normal backoff


# ── Test: Context compaction for repeated admission retries ──────────────

class TestContextCompactionOnAdmissionRetry:
    """Repeated admission retries must compact/prune context before retry."""

    def test_should_compact_after_first_admission_retry(self):
        from agent.provider_failure_recovery import should_compact_before_retry
        assert should_compact_before_retry(attempt=1) is True

    def test_should_not_compact_on_first_attempt(self):
        from agent.provider_failure_recovery import should_compact_before_retry
        assert should_compact_before_retry(attempt=0) is False

    def test_compact_request_reduces_message_count(self):
        from agent.provider_failure_recovery import compact_request_for_retry
        # Messages with tool results and long assistant content should be compacted
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm doing well."},
            {"role": "tool", "content": "x" * 600},
            {"role": "assistant", "content": "A" * 600},
            {"role": "user", "content": "Tell me more"},
            {"role": "assistant", "content": "B" * 600},
            {"role": "user", "content": "And more"},
            {"role": "assistant", "content": "C" * 600},
            {"role": "user", "content": "End"},
            {"role": "assistant", "content": "Done."},
        ]
        compacted = compact_request_for_retry(messages, attempt=2)
        # Should preserve system message and recent messages
        assert compacted[0]["role"] == "system"
        # Old tool results collapsed and long assistant content truncated — check size reduction
        orig_size = sum(len(str(m.get("content", ""))) for m in messages)
        compact_size = sum(len(str(m.get("content", ""))) for m in compacted)
        assert compact_size < orig_size

    def test_compact_request_preserves_system_message(self):
        from agent.provider_failure_recovery import compact_request_for_retry
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        compacted = compact_request_for_retry(messages, attempt=1)
        assert compacted[0]["role"] == "system"
        assert compacted[0]["content"] == "You are helpful."


# ── Test: Duplicate side-effect tool execution prevention ────────────────

class TestDuplicateSideEffectPrevention:
    """Side-effecting tools must not execute twice for the same correlation_id."""

    def test_tool_execution_guard_allows_first_execution(self):
        from agent.provider_failure_recovery import ToolExecutionGuard
        guard = ToolExecutionGuard()
        assert guard.allow_execution("tool_call_123", "terminal_execute") is True

    def test_tool_execution_guard_blocks_duplicate(self):
        from agent.provider_failure_recovery import ToolExecutionGuard
        guard = ToolExecutionGuard()
        assert guard.allow_execution("tool_call_123", "terminal_execute") is True
        assert guard.allow_execution("tool_call_123", "terminal_execute") is False

    def test_different_call_ids_allowed(self):
        from agent.provider_failure_recovery import ToolExecutionGuard
        guard = ToolExecutionGuard()
        assert guard.allow_execution("call_1", "terminal_execute") is True
        assert guard.allow_execution("call_2", "terminal_execute") is True

    def test_non_side_effect_tools_not_guarded(self):
        from agent.provider_failure_recovery import ToolExecutionGuard, SIDE_EFFECTING_TOOLS
        guard = ToolExecutionGuard()
        # read_file is not side-effecting
        assert "read_file" not in SIDE_EFFECTING_TOOLS
        # Even if tracked, non-side-effect tools always allowed
        assert guard.allow_execution("tool_call_1", "read_file") is True
        assert guard.allow_execution("tool_call_1", "read_file") is True


# ── Test: Duplicate final delivery prevention ────────────────────────────

class TestDuplicateDeliveryPrevention:
    """The same final_response must not be delivered twice to the user."""

    def test_delivery_guard_allows_first_delivery(self):
        from agent.provider_failure_recovery import DeliveryGuard
        guard = DeliveryGuard()
        assert guard.allow_delivery("response_abc") is True

    def test_delivery_guard_blocks_duplicate(self):
        from agent.provider_failure_recovery import DeliveryGuard
        guard = DeliveryGuard()
        assert guard.allow_delivery("response_abc") is True
        assert guard.allow_delivery("response_abc") is False

    def test_different_responses_allowed(self):
        from agent.provider_failure_recovery import DeliveryGuard
        guard = DeliveryGuard()
        assert guard.allow_delivery("response_1") is True
        assert guard.allow_delivery("response_2") is True

    def test_idempotency_key_derivation(self):
        from agent.provider_failure_recovery import derive_idempotency_key
        key1 = derive_idempotency_key(turn_id="turn_123", attempt=0)
        key2 = derive_idempotency_key(turn_id="turn_123", attempt=1)
        assert key1 != key2
        assert key1 == derive_idempotency_key(turn_id="turn_123", attempt=0)


# ── Test: Telegram-safe terminal message ────────────────────────────────

class TestTelegramSafeTerminalMessage:
    """On exhaustion, produce a concise Telegram-safe message with no raw
    provider details or secrets."""

    def test_terminal_message_no_secrets(self):
        from agent.provider_failure_recovery import build_terminal_recovery_message
        msg = build_terminal_recovery_message(
            error_class="admission_busy",
            attempts=5,
            window_seconds=120.0,
            provider="nous-portal",
            model="claude-opus-5",
        )
        # Must not leak secrets, keys, or raw provider error details
        assert "api_key" not in msg.lower()
        assert "token" not in msg.lower()
        assert "secret" not in msg.lower()
        assert "bearer" not in msg.lower()
        assert "nous-portal" not in msg  # Provider name not leaked

    def test_terminal_message_states_attempts_and_window(self):
        from agent.provider_failure_recovery import build_terminal_recovery_message
        msg = build_terminal_recovery_message(
            error_class="admission_busy",
            attempts=5,
            window_seconds=120.0,
            provider="test",
            model="test",
        )
        assert "5" in msg  # attempt count
        assert "120" in msg  # window

    def test_terminal_message_suggests_safe_next_step(self):
        from agent.provider_failure_recovery import build_terminal_recovery_message
        msg = build_terminal_recovery_message(
            error_class="admission_busy",
            attempts=5,
            window_seconds=120.0,
            provider="test",
            model="test",
        )
        # Should suggest a safe action
        lower = msg.lower()
        assert "retried" in lower or "new session" in lower or "fresh session" in lower or "later" in lower

    def test_terminal_message_is_short_for_telegram(self):
        from agent.provider_failure_recovery import build_terminal_recovery_message
        msg = build_terminal_recovery_message(
            error_class="admission_busy",
            attempts=5,
            window_seconds=120.0,
            provider="test",
            model="test",
        )
        # Telegram messages should be reasonable length
        assert len(msg) < 500


# ── Test: Kanban integration for terminal exhaustion ────────────────────

class TestKanbanHandoff:
    """On terminal exhaustion, the failure should produce data compatible with
    kanban integration for deterministic handoff/block."""

    def test_terminal_result_has_failure_fields(self):
        from agent.provider_failure_recovery import build_terminal_result
        result = build_terminal_result(
            messages=[],
            api_call_count=3,
            error_class="admission_busy",
            attempts=5,
            window_seconds=120.0,
            provider="test",
            model="test",
        )
        assert result["failed"] is True
        assert result["failure_reason"] == "admission_busy"
        assert result["failure_retryable"] is True
        assert "final_response" in result

    def test_terminal_result_includes_recovery_metadata(self):
        from agent.provider_failure_recovery import build_terminal_result
        result = build_terminal_result(
            messages=[],
            api_call_count=3,
            error_class="admission_busy",
            attempts=5,
            window_seconds=120.0,
            provider="test",
            model="test",
        )
        assert result["recovery_attempts"] == 5
        assert result["recovery_window_seconds"] == 120.0


# ── Test: Retry-After parsing from error ────────────────────────────────

class TestRetryAfterExtraction:
    """Extract Retry-After from error headers or body."""

    def test_extract_from_headers(self):
        from agent.provider_failure_recovery import extract_retry_after
        error = _MockAPIError(
            "busy",
            status_code=429,
            headers={"Retry-After": "30"},
        )
        assert extract_retry_after(error) == 30.0

    def test_extract_from_body(self):
        from agent.provider_failure_recovery import extract_retry_after
        error = _MockAPIError(
            "busy",
            status_code=429,
            body={"retry_after": 45},
        )
        assert extract_retry_after(error) == 45.0

    def test_extract_from_error_obj_in_body(self):
        from agent.provider_failure_recovery import extract_retry_after
        error = _MockAPIError(
            "busy",
            status_code=429,
            body={"error": {"retry_after": 60}},
        )
        assert extract_retry_after(error) == 60.0

    def test_none_when_absent(self):
        from agent.provider_failure_recovery import extract_retry_after
        error = _MockAPIError("busy", status_code=429)
        assert extract_retry_after(error) is None

    def test_capped_at_120(self):
        from agent.provider_failure_recovery import extract_retry_after
        error = _MockAPIError(
            "busy",
            status_code=429,
            headers={"Retry-After": "300"},
        )
        assert extract_retry_after(error) == 120.0
