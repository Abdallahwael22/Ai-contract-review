"""LangSmith tracing helpers.

Tracing is opt-in: it only becomes active when ``LANGSMITH_TRACING=true`` is
set in the environment. Every helper degrades to a no-op when tracing is
disabled, so the worker keeps running locally without a LangSmith account.

Only import this module from activities (never from workflow code) — workflow
code runs inside Temporal's deterministic sandbox and must not perform I/O.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Payloads larger than this are truncated before being sent to LangSmith.
# The full prompt is still sent to the LLM — only the traced copy is trimmed.
MAX_TEXT_CHARS = 20_000

_MAX_DEPTH = 4
_MAX_LIST_ITEMS = 50

_TRUTHY = {"true", "1", "yes", "on"}


# ── Configuration ────────────────────────────────────────────────────────────


def tracing_enabled() -> bool:
    """True when LangSmith tracing is enabled via the environment."""
    value = os.environ.get(
        "LANGSMITH_TRACING",
        os.environ.get("LANGCHAIN_TRACING_V2", ""),
    )
    return value.strip().lower() in _TRUTHY


# ── Payload sanitisation ─────────────────────────────────────────────────────


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}… [truncated {len(text) - max_chars} chars]"


def _sanitize(
    value: Any,
    *,
    drop: frozenset[str] = frozenset(),
    max_chars: int = MAX_TEXT_CHARS,
    depth: int = 0,
) -> Any:
    """Recursively make a value small and JSON-friendly for tracing."""
    if depth > _MAX_DEPTH:
        return "<max depth reached>"

    if isinstance(value, str):
        return _truncate(value, max_chars)

    if isinstance(value, (bytes, bytearray)):
        return f"<{len(value)} bytes>"

    if isinstance(value, (bool, int, float)) or value is None:
        return value

    if isinstance(value, dict):
        return {
            k: _sanitize(v, drop=drop, max_chars=max_chars, depth=depth + 1)
            for k, v in value.items()
            if k not in drop
        }

    if isinstance(value, (list, tuple, set)):
        items = list(value)
        return [
            _sanitize(v, drop=drop, max_chars=max_chars, depth=depth + 1)
            for v in items[:_MAX_LIST_ITEMS]
        ]

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: _sanitize(
                getattr(value, f.name), drop=drop, max_chars=max_chars, depth=depth + 1
            )
            for f in dataclasses.fields(value)
            if f.name not in drop
        }

    return _truncate(repr(value), max_chars)


def _processor(drop: tuple[str, ...], max_chars: int):
    dropped = frozenset(drop)

    def process(value: Any) -> Any:
        return _sanitize(value, drop=dropped, max_chars=max_chars)

    return process


# ── Tracing decorators ───────────────────────────────────────────────────────


def traced(
    name: str,
    *,
    run_type: str = "chain",
    drop_inputs: tuple[str, ...] = (),
    drop_outputs: tuple[str, ...] = (),
    max_chars: int = MAX_TEXT_CHARS,
):
    """Return a ``traceable`` decorator with sanitised, size-capped payloads.

    Apply it to the *implementation* function, not directly to the
    ``@activity.defn`` function: LangSmith's wrapper has a ``*args/**kwargs``
    signature and Temporal rejects that ("Activity cannot have keyword-only
    arguments"). The activity is a thin shell that calls the traced body.
    """
    from langsmith import traceable

    return traceable(
        name=name,
        run_type=run_type,
        process_inputs=_processor(drop_inputs, max_chars),
        process_outputs=_processor(drop_outputs, max_chars),
    )


def annotate_run(metadata: Optional[dict[str, Any]]) -> None:
    """Attach metadata to the current LangSmith run (never raises)."""
    if not metadata:
        return
    try:
        from langsmith.run_helpers import get_current_run_tree

        run = get_current_run_tree()
        if run is not None:
            run.add_metadata(metadata)
    except Exception:  # pragma: no cover - tracing must never break work
        logger.debug("Failed to annotate LangSmith run", exc_info=True)


def current_run_id() -> Optional[str]:
    """ID of the current LangSmith run, or None when tracing is off."""
    try:
        from langsmith.run_helpers import get_current_run_tree

        run = get_current_run_tree()
        return str(run.id) if run is not None else None
    except Exception:  # pragma: no cover - tracing must never break work
        logger.debug("Failed to read current LangSmith run", exc_info=True)
        return None


# ── OpenAI / OpenRouter instrumentation ─────────────────────────────────────


def wrap_openai_client(client: Any) -> Any:
    """Wrap an OpenAI-compatible client so chat calls are logged as LLM runs.

    Works with OpenRouter because it speaks the OpenAI API; the wrapper only
    needs a client that exposes ``chat.completions``.
    """
    if not tracing_enabled():
        return client
    try:
        from langsmith.wrappers import wrap_openai

        return wrap_openai(client, chat_name="OpenRouter Chat")
    except Exception:  # pragma: no cover - tracing must never break work
        logger.warning("Could not wrap OpenAI client for tracing", exc_info=True)
        return client


# ── Feedback & flushing ──────────────────────────────────────────────────────


def record_feedback(
    run_id: Optional[str],
    *,
    decision: str,
    feedback: str = "",
    metadata: Optional[dict[str, Any]] = None,
) -> bool:
    """Attach a human review decision to a run as LangSmith feedback.

    Approve → score 1.0, anything else → score 0.0. Returns True on success.
    Never raises: LangSmith being unavailable must not fail a review.
    """
    if not run_id or not tracing_enabled():
        return False

    try:
        from langsmith.run_trees import get_cached_client

        client = get_cached_client()
        client.create_feedback(
            run_id=run_id,
            key="human_review",
            score=1.0 if decision == "approve" else 0.0,
            comment=feedback or None,
            source_info={"decision": decision, **(metadata or {})},
            # Feedback is cosmetic: never let an outage stall the review.
            stop_after_attempt=2,
        )
        return True
    except Exception:  # pragma: no cover - tracing must never break work
        logger.warning("Failed to record LangSmith feedback", exc_info=True)
        return False


def flush_traces(timeout: float = 10.0) -> None:
    """Flush buffered runs so nothing is lost when the worker shuts down."""
    if not tracing_enabled():
        return
    try:
        from langsmith.run_trees import get_cached_client

        get_cached_client().flush(timeout=timeout)
    except Exception:  # pragma: no cover - tracing must never break shutdown
        logger.debug("Failed to flush LangSmith traces", exc_info=True)
