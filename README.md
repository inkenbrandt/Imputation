# rfr-gapfill

Leakage-safe **Random Forest Robust (RFR)** gap filling for eddy-covariance flux
data, reproducing the method of:

> Zhu, S., Clement, R., McCalmont, J., Davies, C. A., & Hill, T. (2022).
> *Stable gap-filling for longer eddy covariance data gaps: A globally validated
> machine-learning approach for carbon dioxide, water, and energy fluxes.*
> Agricultural and Forest Meteorology, 314, 108777.
> <https://doi.org/10.1016/j.agrformet.2021.108777>

This is an independent implementation, not a copy of the paper's `fluxlib` code.
Its priorities are **scientific reproducibility, explicit assumptions, and
prevention of data leakage** — not maximising predictive scores.

## Status

**Pre-alpha.** The scientific specification is frozen and the package installs
and tests cleanly. The whole path from a raw file to a scored validation run is
implemented: configuration, column mapping, the temporal layer, the
receptive-limiter features, the artificial-gap generator, the model, filling,
metrics, and the validation orchestration, plus an optional FLUXNET2015 adapter.

| Component | State |
|---|---|
| Frozen specification (`docs/`) | done |
| Package scaffold, packaging, CI-ready tests | done |
| Configuration and column-mapping layer | done |
| Timestamp, cadence and elapsed-time utilities | done |
| Metrics, including the all/daytime/nighttime split | done |
| Receptive-limiter features | done |
| Artificial-gap generator | done |
| Model, filling, validation orchestration | done |
| FLUXNET2015 adapter | done |

## Specification first

The implementation is checked against a frozen specification rather than against
prose in the paper:

- [`docs/method_spec.md`](docs/method_spec.md) — the contract: driver lists,
  receptive-limiter features, gap scenario, metrics, and a **Known ambiguities**
  table (A1–A10) covering every point where the paper is silent.
- [`docs/method_spec.yaml`](docs/method_spec.yaml) — machine-readable companion.
  Each block is tagged `provenance: paper` or `provenance: default` so code and
  tests can tell a published fact from a documented choice of ours.
- [`docs/supplement_benchmarks.md`](docs/supplement_benchmarks.md) — which
  supplementary table or figure supports each numerical reproduction target.
- [`docs/fluxnet_adapter.md`](docs/fluxnet_adapter.md) — what the optional
  FLUXNET2015 adapter assumes about that data product, what its QC flags mean,
  and the four naming choices (`F1`–`F4`) it makes on your behalf.

Rule for contributors: **nothing may be labelled "paper exact" unless the article
or its supplements state it.** Where the paper is silent, pick a defensible
default, expose it in configuration, and record it in the ambiguities table.

## Method summary

Two named configurations, both fitted **per site and per target**:

- **RFR3** — Random Forest plus receptive limiter, using the three
  MDS-equivalent drivers: downward shortwave radiation, VPD, air temperature.
- **RFR10** — RFR3 plus net radiation, wind speed, wind direction, soil heat flux,
  soil temperature, relative humidity, soil water content.

The **receptive limiter** is the paper's feature-engineering stage: a radiation
category, elapsed hours since the series start, a hemisphere-aware season tag, and
daily target-flux Q1/Q2/Q3/standard deviation. Setting
`use_receptive_limiter=False` yields the supplement's **ORF** benchmark — same
estimator, same drivers, no engineered features — so the contribution of the
feature engineering is directly measurable.

Daily target statistics are derived from the target itself, which makes them the
main leakage risk. The default `feature_mode="paper_safe"` builds the artificial-gap
mask *before* computing features and derives daily statistics only from
observations visible to the model.

Canonical variable names are internal. FLUXNET2015 column names are a default
mapping, never a hard requirement — bring your own names through a column map:

```python
from rfrgapfill import ColumnMap, RFRConfig

config = RFRConfig(
    mode="RFR3",
    frequency="30min",
    latitude=51.5,                     # or hemisphere="north"; one is required
    random_state=42,
    column_map=ColumnMap({"shortwave": "SW_IN", "vpd": "VPD", "air_temperature": "TA"}),
)
config.to_dict()                       # goes straight into the run manifest
```

`ColumnMap.fluxnet2015("RFR10")` fills in the reference FLUXNET names. Every
scientific choice — radiation thresholds and their boundary convention, the gap
mix and its allocation basis, the daytime threshold, the hyperparameter grid, the
CV strategy, the leakage-safe feature mode — is a configuration field, validated at
construction. Enhancements and the ORF benchmark switch off
`RFRConfig.is_paper_faithful`, so a run can never quietly claim to reproduce the
paper while deviating from it.

## Time is elapsed time, never row position

Gap lengths, the `time_distance_hours` feature and every interval selection are
computed from timestamps: one day is 48 rows only at 30-minute cadence with no
missing rows. `rfrgapfill.time` owns that rule.

```python
from rfrgapfill import prepare_time_index

df, axis = prepare_time_index(df, timestamp="TIMESTAMP_START")   # or a DatetimeIndex

axis.time_step        # inferred from real differences, or declared via frequency=
axis.is_regular       # missing grid points and off-grid stamps reported separately
axis.periods("7d")    # rows a 7-day gap spans at this cadence
axis.elapsed_hours()  # the receptive limiter's time_distance_hours feature
axis.to_dict()        # cadence, coverage and timezone for the run manifest
```

Timestamps must be sorted and unique. Duplicates raise by default;
`on_duplicates="keep_first"|"keep_last"` are the documented preprocessing
options, and neither averages the duplicated rows. Missing rows are *not* an
error — real flux series have them, which is exactly why durations are
elapsed-time quantities.

## Install

Requires Python 3.10+.

```bash
# with uv
uv venv
uv pip install -e ".[dev]"

# or with pip
python -m venv .venv && .venv/Scripts/activate   # .venv/bin/activate on POSIX
pip install -e ".[dev]"
```

Notebooks are not a core or development dependency; install the optional
`notebooks` extra if you want the example notebook environment.

## Filling real gaps

```python
from rfrgapfill import ColumnMap, RFRConfig, RFRGapFiller

config = RFRConfig(
    mode="RFR10",
    frequency="30min",
    latitude=51.5,
    random_state=42,
    column_map=ColumnMap.fluxnet2015("RFR10"),
)

filler = RFRGapFiller(config).fit(df, target="LE", qc_col="LE_QC")
result = filler.fill()

result.frame["LE_filled"]     # observed where measured, predicted where it was not
result.n_filled, result.n_unfilled
filler.manifest()             # environment, config, time axis, training report
```

Observed values are never overwritten in place: filling returns
`<target>_original`, `<target>_filled`, `<target>_is_observed`,
`<target>_is_filled`, `<target>_fill_method` and `<target>_model_version`. A row
missing a required driver is left unfilled rather than filled from an imputed
input, and the input frame is never mutated.

## Reproducing the paper's validation

One call runs the whole artificial-gap experiment: identify the genuinely
measured rows, draw the gap scenario, hide the truth, build leakage-safe
features, tune and fit on what is left, predict inside the gaps, and score.

```python
from rfrgapfill import ColumnMap, validate_rfr

report = validate_rfr(
    df,
    targets=["NEE", "H", "LE"],
    mode="RFR10",
    scenario="zhu2022",              # 25% withheld as 24h/7d/30d gaps, mixed 20/30/50
    latitude=51.5,
    column_map=ColumnMap.fluxnet2015("RFR10"),
    qc_columns={"NEE": "NEE_VUT_REF_QC", "H": "H_F_MDS_QC", "LE": "LE_F_MDS_QC"},
)

report.metrics_frame()               # target x gap class x all/daytime/nighttime
report.gap_manifest                  # start, end, duration, class, observed fraction
report.energy_balance                # measured vs filled EBR over the same gaps
report["NEE"].by_gap_class[GapClass.VERY_LONG].bias_iqr
report.satisfied, report.warnings    # and, when it was not, exactly why not
report.manifest()                    # everything needed to reproduce the run
```

The withheld observations **are** the test set — contiguous temporal intervals,
never a row-wise random holdout, which would leave every held-out half-hour
surrounded by its own neighbours. Given several targets, they are scored on
identical gap locations, as the paper's joint NEE/H/LE validation was.

`compare_receptive_limiter(...)` runs RFR and the ORF benchmark on exactly the
same gaps and returns the paired metrics table.

### One thing the paper does not settle

The daily target statistics are derived from the target, so under a strictly
leakage-safe reading a 30-day gap contains no visible target observation, every
day inside it has no statistics, and **no row inside it is predictable**. The
paper's headline result is 7- and 30-day gaps, so its implementation cannot have
behaved that way; the article does not say what it did instead.

The default is therefore `daily_statistics_strategy="nearest_visible_day"`: a day
with too few visible observations takes the statistics of the closest calendar
day that has enough, ties going to the earlier day. It reads only observations
visible to the model, so it cannot leak, and every substituted day is counted in
the run report. The strict reading is available as `"within_day"`, and what it
costs is measurable — `report[target].coverage` drops to zero for long gaps.
This is ambiguity A4/A4a in [`docs/method_spec.md`](docs/method_spec.md).

## Reading FLUXNET2015 files

The paper ran on FLUXNET2015 FULLSET half-hourly files, so an optional adapter
maps that product onto the generic API. It is the only module that knows FLUXNET
column names; nothing else in the package does.

```python
from rfrgapfill import validate_rfr
from rfrgapfill.fluxnet import inspect_fluxnet, read_fluxnet_csv

frame = read_fluxnet_csv("FLX_XX-Site_FLUXNET2015_FULLSET_HH_2004-2014_1-4.csv")
info = inspect_fluxnet(frame)

info.best_mode                       # RFR10, RFR3, or None if not even RFR3
info.missing_drivers                 # canonical name -> the column that is absent
info.summary()                       # human-readable

report = validate_rfr(
    frame,
    targets=list(info.target_columns()),
    mode=info.best_mode,
    scenario="zhu2022",
    column_map=info.column_map(),     # built from the columns really present
    qc_columns=info.qc_columns(),
    energy_balance_targets=info.heat_targets(),
    latitude=51.5,
)
```

`read_fluxnet_csv` reads a **local** file — the package downloads nothing and
holds no credentials — and fixes the two things that fail silently otherwise:
`YYYYMMDDHHMM` timestamps, and the `-9999` sentinel, which is a
plausible-looking number that would otherwise be trained on.

`inspect_fluxnet` reports availability *before* a fit, because RFR10 is not
available everywhere: the paper's full RFR3/RFR10 comparison used 94 of 194
sites. A site that can only run RFR3 is a result to report, not a problem to
work around by substituting drivers.

On the QC flags, only `0` is a measurement; `1`/`2`/`3` are MDS gap fills of
descending quality. `qc_summary(frame, "LE_F_MDS_QC", target="LE_F_MDS")` shows
what that rule costs at your site before you pay it. The flag semantics, the
depth-indexed soil columns (`SWC_F_MDS_1`), the `TIMESTAMP_START` convention and
the daily-product caveat are all in
[`docs/fluxnet_adapter.md`](docs/fluxnet_adapter.md).

## Development

```bash
pytest              # tests
ruff check .        # lint
ruff format .       # format
mypy                # type check (strict, src/rfrgapfill)
```

Test markers: `slow`, `fluxnet` (needs real FLUXNET2015 input), `supplement`
(needs the journal supplementary files). None of that data is committed here.

## Citing

Cite the paper for the method and [`CITATION.cff`](CITATION.cff) for this
implementation. This package is not affiliated with the authors of Zhu et al.
(2022).

## License

MIT — see [`LICENSE`](LICENSE).
