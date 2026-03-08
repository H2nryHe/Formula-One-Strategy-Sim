"""Shared assumptions metadata for replay-only public-data evaluation."""

from __future__ import annotations

from hashlib import sha256

ASSUMPTIONS_VERSION = "public-data-replay-only-v0"


def default_assumptions_hash() -> str:
    """Return a stable hash tied to the active Stage 0 assumptions profile."""
    return sha256(ASSUMPTIONS_VERSION.encode("utf-8")).hexdigest()
