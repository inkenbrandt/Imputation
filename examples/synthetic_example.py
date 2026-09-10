"""Synthetic end-to-end demonstration.

Builds half-hourly data with known diurnal and seasonal structure, runs the
artificial-gap validation of Zhu et al. (2022) over it -- 24-hour, 7-day and
30-day gaps at a nominal 25% withheld fraction -- and prints the metrics,
the gap manifest and the energy-balance ratio.

The numbers are not the paper's: the site is synthetic, so this demonstrates the
workflow rather than reproducing a benchmark. What it does show is the shape of
a real result, including the weak nighttime skill that aggregate-only reporting
would hide.

Run it with::

    python examples/synthetic_example.py

It takes a couple of minutes; the grid is kept small on purpose.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rfrgapfill import ColumnMap, RFRConfig, validate_rfr

#: Deliberately small, so the example runs in a coffee break rather than an hour.
DEMO_GRID = {"n_estimators": (100,), "min_samples_leaf": (1, 5)}


def synthetic_site(*, days: int = 400, seed: int = 0) -> pd.DataFrame:
    """Return a synthetic half-hourly site with FLUXNET2015 column names.

    Drivers carry a diurnal and a seasonal cycle; the fluxes are smooth functions
    of them plus noise. Twelve percent of each flux is then removed and flagged
    in a ``<target>_QC`` column, standing in for the real gaps every
    eddy-covariance series has before any artificial ones are added.
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range("2019-01-01", periods=days * 48, freq="30min", name="timestamp")
    n = len(index)
    hour = index.hour + index.minute / 60
    day_of_year = index.dayofyear

    shortwave = np.clip(800 * np.sin(np.pi * (hour - 5) / 14), 0, None) * (
        0.7 + 0.3 * np.sin(2 * np.pi * day_of_year / 365)
    )
    shortwave = np.where((hour < 5) | (hour > 19), 0.0, shortwave) + rng.normal(0, 5, n).clip(0)
    air_temperature = (
        12
        + 10 * np.sin(2 * np.pi * (day_of_year - 100) / 365)
        + 5 * np.sin(np.pi * (hour - 6) / 12)
        + rng.normal(0, 1, n)
    )
    humidity = np.clip(
        80 - 0.03 * shortwave - 0.8 * (air_temperature - 12) + rng.normal(0, 4, n), 10, 100
    )
    vpd = np.clip(
        0.61 * np.exp(17.5 * air_temperature / (air_temperature + 241)) * (1 - humidity / 100) * 10,
        0,
        None,
    )
    net_radiation = 0.75 * shortwave - 40 + rng.normal(0, 8, n)
    soil_heat = 0.08 * net_radiation + rng.normal(0, 4, n)
    soil_temperature = (
        12 + 8 * np.sin(2 * np.pi * (day_of_year - 120) / 365) + rng.normal(0, 0.5, n)
    )

    frame = pd.DataFrame(
        {
            "SW_IN_F": shortwave,
            "VPD_F_MDS": vpd,
            "TA_F_MDS": air_temperature,
            "NETRAD": net_radiation,
            "WS": np.clip(2 + rng.normal(0, 1, n), 0.1, None),
            "WD": rng.uniform(0, 360, n),
            "G_F_MDS": soil_heat,
            "TS_F_MDS": soil_temperature,
            "RH": humidity,
            "SWC_F_MDS": np.clip(
                30 + 5 * np.sin(2 * np.pi * (day_of_year - 60) / 365) + rng.normal(0, 1, n), 5, 60
            ),
            "NEE": (
                1.2 * np.exp(0.07 * (soil_temperature - 10))
                - 0.04 * shortwave * np.clip(1 - ((air_temperature - 22) / 18) ** 2, 0, 1)
                + rng.normal(0, 1.2, n)
            ),
            "H": (
                0.30 * (net_radiation - soil_heat)
                + 1.5 * (air_temperature - 12)
                + rng.normal(0, 15, n)
            ),
            "LE": np.clip(
                0.45 * (net_radiation - soil_heat) * (1 - np.exp(-vpd / 8)) + rng.normal(0, 12, n),
                -20,
                None,
            ),
        },
        index=index,
    )
    for target in ("NEE", "H", "LE"):
        flags = np.zeros(n, dtype=int)
        dropped = rng.random(n) < 0.12
        flags[dropped] = 1
        frame.loc[dropped, target] = np.nan
        frame[f"{target}_QC"] = flags
    return frame


def main() -> int:
    """Run the artificial-gap validation on a synthetic site and report it."""
    frame = synthetic_site()
    print(f"synthetic site: {len(frame):,} half-hourly rows, {frame.index[0]} to {frame.index[-1]}")

    report = validate_rfr(
        frame,
        targets=["NEE", "H", "LE"],
        config=RFRConfig(
            mode="RFR10",
            frequency="30min",
            latitude=51.5,
            site_id="SYN-01",
            random_state=42,
            n_jobs=-1,
            column_map=ColumnMap.fluxnet2015("RFR10"),
            hyperparameter_grid=DEMO_GRID,
            cv_folds=3,
        ),
        qc_columns={"NEE": "NEE_QC", "H": "H_QC", "LE": "LE_QC"},
    )

    gaps = report["NEE"].gaps
    print(
        f"\nartificial gaps: {gaps.n_gaps} intervals withholding "
        f"{gaps.achieved_fraction:.1%} of the observations "
        f"(requested {gaps.config.missing_fraction:.0%})"
    )
    print("achieved mix: " + ", ".join(f"{k.value} {v:.0%}" for k, v in gaps.achieved_mix.items()))
    if not report.satisfied:
        print("\nthe requested design was not fully achievable:")
        for reason in report.warnings:
            print(f"  - {reason}")
        print("  read the metrics against the achieved fraction and mix above.")

    print("\nmetrics (target x gap class x subset):")
    print(report.metrics_frame().round(3).to_string(index=False))

    balance = report.energy_balance
    if balance is not None:
        print(
            f"\nenergy-balance ratio over the gaps: measured {balance.measured:.3f}, "
            f"filled {balance.filled:.3f}, difference {balance.difference:+.3f} "
            f"({balance.n_rows:,} rows)"
        )

    print("\nbias IQR by gap class:")
    for target in report.targets:
        parts = [
            f"{gap_class.value} {result.bias_iqr:.3f}"
            for gap_class, result in report[target].by_gap_class.items()
        ]
        print(f"  {target}: " + ", ".join(parts))

    print(
        "\nNighttime skill is far weaker than daytime skill above. That is expected "
        "(see docs/supplement_benchmarks.md) and is why this package never reports "
        "the aggregate alone."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
