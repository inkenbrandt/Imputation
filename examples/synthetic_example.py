"""Synthetic end-to-end demonstration.

Builds half-hourly data with known diurnal and seasonal structure, applies the
artificial-gap scenario (24 h / 7 d / 30 d at a nominal 25% withheld fraction),
fills it, and scores the result -- acceptance tests 34 and 35.

Status: scaffold placeholder. The generator and workflow land with the feature,
gap and validation steps; this script is kept runnable so the example never rots.
"""

from __future__ import annotations

import rfrgapfill


def main() -> int:
    """Report scaffold status until the workflow modules are implemented."""
    print(f"rfr-gapfill {rfrgapfill.__version__} (scaffold)")
    print("Synthetic demonstration is not implemented yet.")
    print("Specification: docs/method_spec.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
