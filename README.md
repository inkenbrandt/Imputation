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
and tests cleanly. Configuration, column mapping and the temporal layer are
implemented and usable; the modelling modules are still placeholders, filled in
step by step. Everything below the "Planned API" heading except `RFRConfig`,
`ColumnMap` and `TimeAxis` is not built yet.

| Component | State |
|---|---|
| Frozen specification (`docs/`) | done |
| Package scaffold, packaging, CI-ready tests | done |
| Configuration and column-mapping layer | done |
| Timestamp, cadence and elapsed-time utilities | done |
| Receptive-limiter features | not started |
| Artificial-gap generator | not started |
| Model, filling, metrics, validation | not started |

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

## Planned API

Not implemented yet — recorded here as the target the modules are built toward.
It is repeated in the specification and will become the tested quick-start once
the modelling steps land.

```python
from rfrgapfill import RFRConfig, RFRGapFiller

config = RFRConfig(mode="RFR10", frequency="30min", hemisphere="north", random_state=42)
filler = RFRGapFiller(config)

filler.fit(df, target="LE", qc_col="LE_QC", column_map={...})
result = filler.fill(df)

report = filler.validate(
    df,
    artificial_gaps=True,
    missing_fraction=0.25,
    gap_mix={"24h": 0.20, "7d": 0.30, "30d": 0.50},
)
```

Observed values are never overwritten in place: filling returns
`<target>_original`, `<target>_filled`, `<target>_is_observed`,
`<target>_is_filled`, `<target>_fill_method` and `<target>_model_version`.

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
