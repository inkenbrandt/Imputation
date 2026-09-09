"""Timestamp handling, cadence inference and elapsed-time utilities.

Validates monotonic, unique timestamps, infers or accepts the time step, and
computes gap lengths from elapsed time rather than row counts.

Note: this module shadows the stdlib ``time`` only inside the package
namespace; absolute imports elsewhere are unaffected.

Status: scaffold placeholder; implemented in Step 4.
"""

from __future__ import annotations

__all__: list[str] = []
