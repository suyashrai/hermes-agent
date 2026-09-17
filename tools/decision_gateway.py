"""Gateway-native three-option decision prompt system (Yes/No/Custom text).

Implements a pending decision registry keyed by decision_id and session_key,
with support for:
- Yes/No selection with immediate resolution
- Custom text collection via text interception
- Authorization validation (chat/thread/sender)
- First-writer-wins atomic claim on resolution
- Process-local state (lost on gateway restart)

Matches clarify_gateway.py patterns but with three-option semantics
instead of generic multi-choice + other.
"""

import asyncio
import logging
import threading
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class DecisionEntry:
    decision_id: str
    session_key: str
    question: str
    chat_id: str
    thread_id: Optional[str]
    sender_id: str
    sender_name: str
    created_at: float = field(default_factory=time.time)
    resolved: bool = False
    response: Optional[str] = None  # "yes", "no", or custom text
    awaiting_text: bool = False
    timeout_seconds: Optional[float] = None
    expiry_time: Optional[float] = None


class DecisionRegistry:
    """Thread-safe registry for pending three-option decisions."""

    def __init__(self):
        self._lock = threading.RLock()
        self._entries: Dict[str, DecisionEntry] = {}
        self._session_index: Dict[str, List[str]] = {}
        self._resolution_callbacks: Dict[str, asyncio.Future] = {}
        self._text_awaiting_callbacks: Dict[str, asyncio.Future] = {}
        self._timeout_tasks: Dict[str, asyncio.Task] = {}

    def register(
        self,
        decision_id: str,
        session_key: str,
        question: str,
        chat_id: str,
        thread_id: Optional[str],
        sender_id: str,
        sender_name: str,
        timeout_seconds: Optional[float] = None,
    ) -> None:
        """Register a new pending decision.

        Args:
            decision_id: Unique identifier for the decision
            session_key: Session identifier (for text interception)
            question: The prompt text to display
            chat_id: Chat where the decision was presented
            thread_id: Optional thread identifier (for Telegram groups)
            sender_id: User ID of the founder/user
            sender_name: Display name of the sender
            timeout_seconds: Optional timeout (0 or None = infinite)
        """
        with self._lock:
            if decision_id in self._entries:
                raise ValueError(f"Decision already registered: {decision_id}")

            entry = DecisionEntry(
                decision_id=decision_id,
                session_key=session_key,
                question=question,
                chat_id=chat_id,
                thread_id=thread_id,
                sender_id=sender_id,
                sender_name=sender_name,
                timeout_seconds=timeout_seconds,
            )

            if timeout_seconds and timeout_seconds > 0:
                entry.expiry_time = entry.created_at + timeout_seconds
                asyncio.create_task(self._expire_decision(decision_id, timeout_seconds))

            self._entries[decision_id] = entry
            self._session_index.setdefault(session_key, []).append(decision_id)

            logger.info(
                "Registered decision: %s for session %s (chat: %s)",
                decision_id,
                session_key,
                chat_id,
            )

    def resolve(
        self,
        decision_id: str,
        response: str,
        authorized: bool = True,
    ) -> bool:
        """Resolve a decision.

        Args:
            decision_id: ID of decision to resolve
            response: Response value ("yes", "no", or custom text)
            authorized: Whether the resolution is authorized

        Returns:
            True if decision was found and resolved, False otherwise
        """
        with self._lock:
            entry = self._entries.get(decision_id)
            if not entry:
                logger.debug("Decision not found for resolution: %s", decision_id)
                return False

            if entry.resolved:
                logger.debug("Decision already resolved: %s", decision_id)
                return False

            if not authorized:
                logger.warning(
                    "Unauthorized decision resolution attempt: %s from %s",
                    decision_id,
                    entry.sender_id,
                )
                return False

            entry.resolved = True
            entry.response = response

            # Complete any waiters
            fut = self._resolution_callbacks.pop(decision_id, None)
            if fut and not fut.done():
                fut.set_result(response)

            # Clean up timeout task if exists
            if decision_id in self._timeout_tasks:
                self._timeout_tasks[decision_id].cancel()
                del self._timeout_tasks[decision_id]

            logger.info(
                "Decision resolved: %s with response: %s (session: %s)",
                decision_id,
                response[:50] if len(response) > 50 else response,
                entry.session_key,
            )

            return True

    def mark_awaiting_text(self, decision_id: str) -> bool:
        """Mark a decision as awaiting custom text response.

        Args:
            decision_id: ID of decision to mark

        Returns:
            True if decision was found and marked, False otherwise
        """
        with self._lock:
            entry = self._entries.get(decision_id)
            if not entry or entry.resolved:
                return False

            entry.awaiting_text = True
            return True

    def attempt_text_response(self, session_key: str) -> Tuple[bool, Optional[str]]:
        """Attempt to find a pending decision awaiting text response for session.

        Args:
            session_key: Session to look for awaiting text decision

        Returns:
            Tuple of (found: bool, decision_id: Optional[str])
            If found and awaiting text, returns the decision_id and clears its state.
        """
        with self._lock:
            decision_ids = self._session_index.get(session_key, [])
            for decision_id in reversed(decision_ids):
                entry = self._entries.get(decision_id)
                if entry and entry.awaiting_text and not entry.resolved:
                    entry.awaiting_text = False
                    return True, decision_id
            return False, None

    def get_pending_for_session(self, session_key: str) -> List[str]:
        """Get all pending decision IDs for a session."""
        with self._lock:
            return list(self._session_index.get(session_key, []))

    def clear_session(self, session_key: str) -> int:
        """Clear all pending decisions for a session.

        Args:
            session_key: Session to clear

        Returns:
            Number of decisions cleared
        """
        with self._lock:
            decision_ids = self._session_index.pop(session_key, [])
            for decision_id in decision_ids:
                entry = self._entries.pop(decision_id, None)
                if entry:
                    if decision_id in self._timeout_tasks:
                        self._timeout_tasks[decision_id].cancel()
                        del self._timeout_tasks[decision_id]

                    # Complete waiters with None (timeout/clear)
                    fut = self._resolution_callbacks.pop(decision_id, None)
                    if fut and not fut.done():
                        fut.set_result(None)

            logger.info("Cleared %d pending decisions for session: %s", len(decision_ids), session_key)
            return len(decision_ids)

    def get_entry(self, decision_id: str) -> Optional[DecisionEntry]:
        """Get a decision entry by ID (for inspection)."""
        with self._lock:
            return self._entries.get(decision_id)

    async def _expire_decision(self, decision_id: str, timeout_seconds: float):
        """Background task to expire a decision after timeout."""
        try:
            await asyncio.sleep(timeout_seconds)
            with self._lock:
                entry = self._entries.get(decision_id)
                if entry and not entry.resolved:
                    logger.info("Decision expired due to timeout: %s", decision_id)
                    entry.resolved = True
                    entry.response = None

                    # Complete any waiters
                    fut = self._resolution_callbacks.pop(decision_id, None)
                    if fut and not fut.done():
                        fut.set_result(None)

                    # Remove from session index
                    if entry.session_key in self._session_index:
                        try:
                            self._session_index[entry.session_key].remove(decision_id)
                            if not self._session_index[entry.session_key]:
                                del self._session_index[entry.session_key]
                        except (ValueError, KeyError):
                            pass

                    self._entries.pop(decision_id, None)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("Error expiring decision %s: %s", decision_id, e)


# Global registry instance
_registry = DecisionRegistry()


def register_decision(
    decision_id: str,
    session_key: str,
    question: str,
    chat_id: str,
    thread_id: Optional[str],
    sender_id: str,
    sender_name: str,
    timeout_seconds: Optional[float] = None,
) -> None:
    """Public API: Register a new pending decision."""
    _registry.register(
        decision_id=decision_id,
        session_key=session_key,
        question=question,
        chat_id=chat_id,
        thread_id=thread_id,
        sender_id=sender_id,
        sender_name=sender_name,
        timeout_seconds=timeout_seconds,
    )


def wait_for_response(decision_id: str) -> str:
    """Public API: Wait for a decision to be resolved.

    Returns "yes", "no", custom text, or None on timeout.
    """
    loop = asyncio.get_event_loop()
    fut = loop.create_future()

    with _registry._lock:
        _registry._resolution_callbacks[decision_id] = fut

    return fut.result()


def resolve_gateway_decision(decision_id: str, response: str) -> bool:
    """Public API: Resolve a decision.

    Called by callback handlers to fulfill a decision.
    Returns True if decision was resolved.
    """
    return _registry.resolve(decision_id, response, authorized=True)


def get_pending_for_session(session_key: str) -> List[str]:
    """Public API: Get pending decision IDs for session."""
    return _registry.get_pending_for_session(session_key)


def clear_session(session_key: str) -> None:
    """Public API: Clear all pending decisions for a session."""
    _registry.clear_session(session_key)


def mark_awaiting_text(decision_id: str) -> bool:
    """Public API: Mark a decision as awaiting custom text."""
    return _registry.mark_awaiting_text(decision_id)


def attempt_text_response_for_session(session_key: str) -> Tuple[bool, Optional[str]]:
    """Public API: Attempt to get text response for session.

    Returns (found: bool, decision_id: Optional[str])
    """
    return _registry.attempt_text_response(session_key)


def get_decision_entry(decision_id: str) -> Optional[DecisionEntry]:
    """Public API: Get a decision entry for inspection."""
    return _registry.get_entry(decision_id)