"""Synthetic end-to-end demonstration: RFR3 and RFR10, validated and scored.

Step 17 of the implementation plan, as a runnable script. Builds half-hourly data
with known diurnal and seasonal structure, places the paper's 24 h / 7 d / 30 d
artificial gaps over it, and runs **both** published configurations across NEE, H
and LE on exactly the same withheld intervals, reporting for each:

* R2, slope, RMSE and bias (method_spec.md 6.1);
* the same four over daytime and nighttime rows (6.2);
* the same four per gap class, with the spread of bias across gaps (6.3);
* the energy-balance ratio, measured against filled (6.4);
* the gap manifest the scenario placed (4.4);
* both arms lined up against gap duration, with the published Table S3
  medians printed beside them for reference (6.5).

and checks the property the whole design rests on: **no observed value changed**.

Nothing here asserts that RFR10 beats RFR3. On one synthetic site that would not
be a scientific claim, and the paper's own evidence for it is a distribution
across 94 sites (see ``docs/supplement_benchmarks.md``).

Run it with::

    python examples/synthetic_example.py                 # a fast, small grid
    python examples/synthetic_example.py --full-grid     # the package default
    python examples/synthetic_example.py --manifests out # write run manifests

Nothing is downloaded: the site is generated from a seed (Step 16).
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from rfrgapfill import (
    DEFAULT_HYPERPARAMETER_GRID,
    FeatureConfig,
    SyntheticSite,
    ValidationReport,
    benchmark_table,
    gap_length_pivot,
    gap_length_table,
    synthetic_site,
    validate_rfr,
)

#: Enough trees to be a Random Forest, few enough to run in a coffee break. The
#: package default grid (A1) is what ``--full-grid`` uses instead.
QUICK_GRID = {"n_estimators": (100,)}

#: A 7-day gap covers whole calendar days, so under the documented default
#: ``daily_statistic_strategy="missing"`` it has no daily target statistics and
#: receives no predictions at all (ambiguity A4). A long-gap run has to choose a
#: reaching strategy deliberately; this is that choice, and it is recorded in
#: every manifest the run writes.
LONG_GAP_STRATEGY = "rolling_available"

TARGETS = ("NEE", "H", "LE")


def build_site(days: int, seed: int) -> SyntheticSite:
    """Return the synthetic site the demonstration runs on."""
    return synthetic_site(days=days, seed=seed)


def run_arm(
    site: SyntheticSite, mode: str, *, full_grid: bool, gaps, jobs: int | None
) -> ValidationReport:
    """Validate every target under one driver set, over the given intervals.

    ``jobs`` is passed to the forest, not to the grid search, so one setting
    cannot multiply into folds x candidates x trees workers. It changes how long
    the run takes and nothing about what it produces: the forest is seeded.
    """
    config = site.config(
        mode,
        features=FeatureConfig(daily_statistic_strategy=LONG_GAP_STRATEGY),
        hyperparameter_grid=DEFAULT_HYPERPARAMETER_GRID if full_grid else QUICK_GRID,
        n_jobs=jobs,
    )
    return validate_rfr(
        site.frame,
        config=config,
        targets=list(TARGETS),
        qc_columns=dict(site.qc_columns()),
        gaps=gaps,
    )


def show_gap_manifest(report: ValidationReport) -> None:
    """Print the placed intervals and what the scenario achieved."""
    print("\n== Gap manifest ==")
    print(report.gaps.summary())
    frame = report.gaps.to_frame()
    print(frame.to_string(index=False))


def show_metrics(report: ValidationReport) -> None:
    """Print the core metrics by target, gap class and day/night subset."""
    print(f"\n== {report.method}: metrics ==")
    table = report.to_frame().round({"r2": 3, "slope": 3, "rmse": 3, "bias": 3})
    print(table.to_string(index=False))

    print(f"\n== {report.method}: bias spread across the gaps of each class ==")
    rows = [
        {"target": result.target, "gap_class": gap_class.value, **spread.to_dict()}
        for result in report
        for gap_class, spread in result.bias_spread.items()
    ]
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    if report.energy_balance is not None:
        balance = report.energy_balance
        print(f"\n== {report.method}: energy-balance ratio (H + LE) ==")
        print(
            f"rows={balance.n}  measured={balance.measured:.4f}  "
            f"filled={balance.filled:.4f}  difference={balance.difference:+.4f}"
        )


def show_gap_length_comparison(reports: Mapping[str, ValidationReport]) -> None:
    """Print both arms against gap duration, and the published medians beside them."""
    table = gap_length_table(list(reports.values()), site="synthetic")
    print("\n== Skill by gap length: the arms side by side (Step 18) ==")
    for metric in ("r2", "rmse"):
        print(f"\n{metric}, all observations:")
        print(gap_length_pivot(table, metric=metric).round(3).to_string())

    print("\n== The published medians, for reference only ==")
    print(gap_length_pivot(benchmark_table(), metric="r2", gap_class="all").round(2).to_string())
    print(
        "\nThose are medians across the paper's 94 FLUXNET sites "
        "(docs/supplement_benchmarks.md), not a target this synthetic\n"
        "site is expected to reach, and no comparison against them is made or "
        "implied here. A real reproduction run over matching\n"
        "FLUXNET inputs would aggregate its sites with median_across_sites() and "
        "call compare_to_benchmarks(), which refuses to\n"
        "difference NEE RMSE or bias until the units are converted explicitly "
        "(ambiguity A10)."
    )


def check_observations_untouched(
    site: SyntheticSite, before: pd.DataFrame, report: ValidationReport
) -> None:
    """Verify the run left every measured value exactly as it found it."""
    print(f"\n== {report.method}: observed values ==")
    pd.testing.assert_frame_equal(site.frame, before)
    print("input frame unchanged: yes")
    for result in report:
        truth = result.features.truth
        original = before[result.target].astype(float)
        pd.testing.assert_series_equal(truth, original, check_names=False)
        outside = ~result.features.holdout_mask.to_numpy()
        predicted_outside = int(result.predictions.notna().to_numpy()[outside].sum())
        print(
            f"  {result.target}: truth identical to input; "
            f"{predicted_outside} prediction(s) outside the artificial gaps"
        )


def write_manifests(report: ValidationReport, directory: Path) -> None:
    """Write one run manifest per target, each checked for completeness."""
    directory.mkdir(parents=True, exist_ok=True)
    for result in report:
        path = directory / f"{report.method}_{result.target}_run.json"
        report.manifest(result.target).save(path)
        print(f"  wrote {path}")


def main() -> int:
    """Run both published configurations and report what they produced."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--days", type=int, default=365, help="length of the record in days")
    parser.add_argument("--seed", type=int, default=20220101, help="seed for the whole fixture")
    parser.add_argument(
        "--full-grid",
        action="store_true",
        help="search the package's documented default grid instead of a single point",
    )
    parser.add_argument(
        "--manifests", type=Path, default=None, help="directory to write run manifests into"
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=-1,
        help="n_jobs for the forest (-1 uses every core; results are unaffected)",
    )
    args = parser.parse_args()

    site = build_site(args.days, args.seed)
    before = site.frame.copy(deep=True)
    print(
        f"site {site.site_id}: {len(site.frame)} rows at {site.time_step}, "
        f"latitude {site.latitude} ({site.hemisphere.value}ern hemisphere)"
    )

    # One set of intervals for both arms and all three fluxes, so every number
    # below is comparable to every other (method_spec.md 4.4).
    gaps = site.known_gaps()

    reports = {}
    for mode in ("RFR3", "RFR10"):
        print(f"\nrunning {mode} on {', '.join(TARGETS)} ...")
        reports[mode] = run_arm(site, mode, full_grid=args.full_grid, gaps=gaps, jobs=args.jobs)

    show_gap_manifest(reports["RFR3"])
    for mode, report in reports.items():
        show_metrics(report)
        check_observations_untouched(site, before, report)
        if args.manifests is not None:
            print(f"\n== {mode}: run manifests ==")
            write_manifests(report, args.manifests)

    show_gap_length_comparison(reports)

    print(
        "\nBoth configurations completed. No metric is asserted to favour either arm: "
        "one synthetic site is a software check, not evidence about the method."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
