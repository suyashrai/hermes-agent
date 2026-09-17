"""Deterministic provider-failure recovery for Telegram-facing agents.

Handles transient provider failures (admission-busy, timeout, connection,
retryable 5xx) with bounded recovery: max 5 attempts within 120 seconds,
exponential backoff honoring Retry-After. Prevents duplicate tool execution
and duplicate user-visible delivery. Produces Telegram-safe terminal messages.

Design invariants:
- No secrets or raw provider error details in user-facing messages.
- Context compaction before retry on repeated admission retries.
- Idempotency keys derive from turn_id + attempt for correlation.
- Side-effecting tool calls are guarded by a call-id dedup set.
- Final delivery is guarded by response dedup to prevent duplicates.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────

PROVIDER_RECOVERY_MAX_ATTEMPTS = 5
PROVIDER_RECOVERY_WINDOW_SECONDS = 120.0

_PROVIDER_RECOVERY_BASE_BACKOFF = 2.0  # seconds
_PROVIDER_RECOVERY_MAX_BACKOFF = 60.0  # seconds
_PROVIDER_RECOVERY_BACKOFF_JITTER = 0.25  # ratio
_PROVIDER_RECOVERY_RETRY_AFTER_CAP = 120.0  # seconds

# Side-effecting tool names that must not execute twice for the same call_id.
SIDE_EFFECTING_TOOLS = frozenset({
    "terminal_execute", "terminal", "bash", "shell",
    "write_file", "patch", "create_file",
    "webhook_trigger", "send_notification",
    "cronjob", "cron_add", "cron_remove",
    "git_commit", "git_push",
})

# Maximum message age to keep when compacting (most recent N messages).
_COMPACT_KEEP_RECENT = 6
_COMPACT_SYSTEM_PLACEHOLDER = (
    "[Context compacted for retry: earlier conversation history pruned to reduce request size]"
)


# ── Recovery state ──────────────────────────────────────────────────────

@dataclass
class ProviderRecoveryState:
    """Tracks per-request provider failure recovery attempts within the
    bounded time window. Reset on a new original request."""

    attempts: int = 0
    start_time: Optional[float] = None  # monotonic
    error_class: Optional[str] = None

    def start(self, now: float, error_class: str) -> None:
        self.start_time = now
        self.error_class = error_class

    def increment(self) -> None:
        self.attempts += 1

    def is_exhausted(self) -> bool:
        return self.attempts >= PROVIDER_RECOVERY_MAX_ATTEMPTS

    def window_expired(self, now: float) -> bool:
        if self.start_time is None:
            return False
        return (now - self.start_time) >= PROVIDER_RECOVERY_WINDOW_SECONDS

    def is_same_error_class(self, error_class: str) -> bool:
        return self.error_class == error_class

    def reset(self) -> None:
        self.attempts = 0
        self.start_time = None
        self.error_class = None


# ── Backoff ─────────────────────────────────────────────────────────────

def compute_recovery_backoff(
    attempt: int,
    *,
    retry_after: Optional[float] = None,
) -> float:
    """Bounded exponential backoff for provider recovery retries.

    ``attempt`` is 0-based (attempt 0 = first retry). Returns seconds to wait.
    Honors Retry-After header when present, capped at 120 seconds.
    """
    if retry_after is not None and retry_after > 0:
        return min(retry_after, _PROVIDER_RECOVERY_RETRY_AFTER_CAP)

    base = _PROVIDER_RECOVERY_BASE_BACKOFF
    delay = min(base * (2 ** attempt), _PROVIDER_RECOVERY_MAX_BACKOFF)
    # Add jitter
    import random
    jitter = random.uniform(0, _PROVIDER_RECOVERY_BACKOFF_JITTER * delay)
    return delay + jitter


# ── Retry-After extraction ──────────────────────────────────────────────

def extract_retry_after(error: Any) -> Optional[float]:
    """Extract Retry-After from error headers or body, capped at 120s."""
    # Check headers
    headers = getattr(error, "headers", None)
    if headers and hasattr(headers, "get"):
        val = headers.get("Retry-After") or headers.get("retry-after")
        if val is not None:
            try:
                return min(float(val), _PROVIDER_RECOVERY_RETRY_AFTER_CAP)
            except (TypeError, ValueError):
                pass
    # Check response.headers
    response = getattr(error, "response", None)
    if response is not None:
        resp_headers = getattr(response, "headers", None)
        if resp_headers and hasattr(resp_headers, "get"):
            val = resp_headers.get("Retry-After") or resp_headers.get("retry-after")
            if val is not None:
                try:
                    return min(float(val), _PROVIDER_RECOVERY_RETRY_AFTER_CAP)
                except (TypeError, ValueError):
                    pass
    # Check body
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        val = body.get("retry_after")
        if val is None:
            error_obj = body.get("error")
            if isinstance(error_obj, dict):
                val = error_obj.get("retry_after")
        if val is not None:
            try:
                return min(float(val), _PROVIDER_RECOVERY_RETRY_AFTER_CAP)
            except (TypeError, ValueError):
                pass
    return None


# ── Context compaction ──────────────────────────────────────────────────

def should_compact_before_retry(attempt: int) -> bool:
    """True when context should be compacted before retry (after first admission retry)."""
    return attempt >= 1


def compact_request_for_retry(
    messages: List[Dict[str, Any]],
    attempt: int,
) -> List[Dict[str, Any]]:
    """Compact messages for retry by summarizing old tool results and
    pruning old assistant content while preserving the system message
    and recent conversation.

    Repeated admission retries must compact/prune context rather than
    resend oversized context unchanged.
    """
    if not messages:
        return messages

    result = []

    # Always keep the system message (first item if it's a system role).
    if messages[0].get("role") == "system":
        result.append(messages[0])
        remaining = messages[1:]
    else:
        remaining = messages

    if not remaining:
        return result

    # Keep the most recent messages intact.
    keep = min(_COMPACT_KEEP_RECENT, len(remaining))
    recent = remaining[-keep:]
    old = remaining[:-keep]

    # Summarize old messages: compress tool results and long assistant messages.
    tool_result_marker_added = False
    for msg in old:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "tool" or (role == "assistant" and msg.get("tool_calls")):
            # Collapse consecutive tool results into a single marker.
            if not tool_result_marker_added:
                result.append({
                    "role": "user",
                    "content": "[tool results omitted for context reduction]",
                })
                tool_result_marker_added = True
        elif role == "assistant" and isinstance(content, str) and len(content) > 500:
            # Truncate long assistant messages.
            result.append({
                "role": "assistant",
                "content": content[:200] + "... [truncated for context reduction]",
            })
        else:
            result.append(msg)

    result.extend(recent)
    return result


# ── Duplicate side-effect prevention ────────────────────────────────────

class ToolExecutionGuard:
    """Prevents duplicate execution of side-effecting tools.

    Tracks call_ids that have been executed; a duplicate call_id for the
    same side-effecting tool is blocked.
    """

    def __init__(self) -> None:
        self._executed: set = set()

    def allow_execution(self, call_id: str, tool_name: str) -> bool:
        """Returns True if execution is allowed (not a duplicate side-effect)."""
        if tool_name not in SIDE_EFFECTING_TOOLS:
            return True
        key = f"{call_id}:{tool_name}"
        if key in self._executed:
            return False
        self._executed.add(key)
        return True


# ── Duplicate delivery prevention ───────────────────────────────────────

class DeliveryGuard:
    """Prevents duplicate user-visible delivery of the same response content."""

    def __init__(self) -> None:
        self._delivered: set = set()

    def allow_delivery(self, response_content: str) -> bool:
        """Returns True if this response hasn't been delivered yet."""
        # Use a hash to avoid storing large strings in the set.
        h = hashlib.sha256(response_content.encode("utf-8", errors="replace")).hexdigest()[:32]
        if h in self._delivered:
            return False
        self._delivered.add(h)
        return True


def derive_idempotency_key(turn_id: str, attempt: int) -> str:
    """Derive a deterministic idempotency key for a recovery attempt."""
    return f"{turn_id}:attempt:{attempt}"


# ── Terminal message ────────────────────────────────────────────────────

def build_terminal_recovery_message(
    *,
    error_class: str,
    attempts: int,
    window_seconds: float,
    provider: str,
    model: str,
) -> str:
    """Build a Telegram-safe concise terminal message on recovery exhaustion.

    No raw provider details, no secrets, no API keys. Safe next step included.
    """
    return (
        f"Request retried {attempts} times within {int(window_seconds)} seconds "
        f"and failed with persistent {error_class} errors. "
        f"Try sending a new message in a few minutes, or start a fresh session if the issue persists."
    )


def build_terminal_result(
    *,
    messages: List[Dict[str, Any]],
    api_call_count: int,
    error_class: str,
    attempts: int,
    window_seconds: float,
    provider: str,
    model: str,
) -> Dict[str, Any]:
    """Build a terminal turn result dict for provider-failure recovery exhaustion.

    Compatible with existing kanban integration and failure reporting.
    """
    final_response = build_terminal_recovery_message(
        error_class=error_class,
        attempts=attempts,
        window_seconds=window_seconds,
        provider=provider,
        model=model,
    )
    return {
        "completed": False,
        "failed": True,
        "partial": True,
        "final_response": final_response,
        "messages": messages,
        "api_calls": api_call_count,
        "error": final_response,
        "failure_reason": error_class,
        "failure_retryable": True,
        "recovery_attempts": attempts,
        "recovery_window_seconds": window_seconds,
    }
