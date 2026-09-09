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
and tests cleanly. Configuration, column mapping, the temporal layer, the
receptive-limiter features, the leakage-safe validation feature workflow, the
Random Forest and the operational fill API are implemented and usable; the
artificial-gap generator, the metrics and the paper-validation workflow are still
placeholders, filled in step by step.

| Component | State |
|---|---|
| Frozen specification (`docs/`) | done |
| Package scaffold, packaging, CI-ready tests | done |
| Configuration and column-mapping layer | done |
| Timestamp, cadence and elapsed-time utilities | done |
| Receptive-limiter features, ORF pairing | done |
| Leakage-safe validation features | done |
| Random Forest fitting, tuning and persistence | done |
| Operational fill API and provenance | done |
| Artificial-gap generator | not started |
| Metrics and paper-validation workflow | not started |

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

## Leakage-safe validation features

The daily target statistics are built *from the target*, so an artificial-gap
validation that computes them naively lets the hidden truth build its own
predictors. `rfrgapfill.leakage` is the workflow that prevents it, and validation
features should be built through it rather than by calling the transformers
directly.

```python
from rfrgapfill import build_validation_features, holdout_mask_from_intervals

holdout = holdout_mask_from_intervals(df.index, [("2020-06-06", "2020-06-13")])

validation = build_validation_features(
    df, config=config, target="LE", holdout=holdout, qc_column="LE_QC"
)

validation.training_features(), validation.training_target()  # what the model sees
validation.holdout_features()                                 # what it predicts
validation.scoring_truth()                                    # scoring only
validation.to_dict()                                          # for the run manifest
```

The truth is a separate attribute from the features, and two independent
protections keep it out of them: the held-out values are removed from the frame
the features are read from, *and* the transformer is told which rows are visible.
Either alone is sufficient, and both are checked:

```python
from rfrgapfill import detect_target_leakage, require_no_target_leakage

detect_target_leakage(df, config=config, target="LE", holdout=holdout)   # () when safe
require_no_target_leakage(df, config=config, target="LE", holdout=holdout)
```

The probe replaces the hidden truth with absurd values, rebuilds every feature —
with and without the frame-level masking — and reports any column that moved. In
`paper_safe` mode it must report nothing, for every daily-statistic strategy.

One consequence worth knowing before a long-gap run: under the documented default
`daily_statistic_strategy="missing"`, a gap covering a whole calendar day leaves
every row of that day without daily statistics, so 7-day and 30-day gaps produce
no complete feature rows at all. The paper does not say what it did here
(ambiguity A4), so the alternatives are explicit — `within_day_available`,
`neighbor_day_fallback`, `rolling_available` — all of them drawing only on visible
observations, and `to_dict()` reports `holdout_rows_with_complete_features` so the
choice cannot go unnoticed.

## The Random Forest itself

`RFRModel` is the low-level model layer: a `RandomForestRegressor` inside a
`GridSearchCV` whose grid, folds, seed and `n_jobs` all come from the
configuration. It takes a feature matrix and a target vector and nothing else, so
the same class serves an RFR run, the ORF benchmark and an operational fill.

```python
from rfrgapfill import RFRModel, build_feature_matrix

features = build_feature_matrix(df, config=config, target="LE")

model = RFRModel(config, target="LE").fit(features, df["LE"])
model.get_feature_names()      # the order the model was fitted on
model.get_best_params()        # what GridSearchCV chose from the configured grid
model.fit_report.to_dict()     # rows offered, fitted, and why the rest were not
model.to_dict()                # grid, folds, seed, best params, versions

predictions = model.predict(features)          # a float Series on df's index
model.save("LE.joblib")
RFRModel.load("LE.joblib")                     # config revalidated as it loads
```

The grid is **ours**, not the paper's: the article says `GridSearchCV` was used and
never enumerates the search, so `hyperparameter_grid_is_paper_exact` is `False` in
every manifest this package writes (ambiguity A1).

A row missing any predictor is dropped at fit and left `NaN` at prediction —
recent scikit-learn forests would accept `NaN` and quietly impute, which is not a
rule the paper documents. `predict(..., on_incomplete="raise")` is the fail-loudly
alternative, and `fit_report` says exactly how many rows went and which feature
took them.

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

`RFRGapFiller` is the operational interface: one site, one target, fit then fill.

```python
from rfrgapfill import ColumnMap, RFRConfig, RFRGapFiller

config = RFRConfig(
    mode="RFR10",
    frequency="30min",
    hemisphere="north",
    random_state=42,
    column_map=ColumnMap.fluxnet2015("RFR10"),
)

filler = RFRGapFiller(config).fit(df, target="LE", qc_column="LE_QC")
result = filler.fill(df)

result.filled              # observed where observed, predicted in the gaps
result.report.summary()    # "LE: filled 812 of 812 candidate row(s) with RFR10."
filler.to_dict()           # the run manifest: config, columns, features, model
```

`result.frame` is a **new** frame — your DataFrame is never modified — carrying
its own columns plus the six provenance columns of `docs/method_spec.md`
section 7:

| Column | Meaning |
|---|---|
| `LE_original` | the target exactly as it arrived |
| `LE_filled` | the best available series |
| `LE_is_observed` | a genuine, QC-accepted measurement |
| `LE_is_filled` | this package predicted this value |
| `LE_fill_method` | `observed`, `pre_filled`, `unfilled_incomplete_features`, or the arm (`RFR10`) |
| `LE_model_version` | which fitted model produced a filled value |

The rules the class enforces, none of them optional:

- **only observed rows train.** A value the QC flag marks as gap-filled before
  ingestion is neither trained on nor used to compute the daily statistics.
- **a present value is never replaced.** Only gaps are predicted. Pre-filled
  values are carried through and labelled; `fill(df, refill_pre_filled=True)` is
  the explicit opt-in, and the way to run RFR over a FLUXNET target column that
  already arrived complete.
- **a row lacking a predictor is not filled.** It stays missing and is counted, or
  `fill(df, on_incomplete="raise")` fails instead. Nothing is fabricated.
- **`time_distance_hours` keeps the origin it was fitted with**, so filling a
  later slice of the same series does not restart the clock.

One trap worth knowing, and the report names it for you: under the default
`daily_statistic_strategy="missing"` a gap covering a whole calendar day has no
daily statistics, so **none of its rows can be filled** (ambiguity A4). The fill
warns rather than handing back a quietly empty gap, and `report.worst_feature`
points at the daily statistic responsible. Choose a reaching strategy for
multi-day gaps:

```python
config = config.replace(
    features=config.features.replace(
        daily_statistic_strategy="rolling_available", fallback_window_days=7
    )
)
```

## Planned API

Still to land: the artificial-gap generator, the metrics and the paper-validation
workflow that ties them together.

```python
report = filler.validate(
    df,
    artificial_gaps=True,
    missing_fraction=0.25,
    gap_mix={"24h": 0.20, "7d": 0.30, "30d": 0.50},
)
```

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
