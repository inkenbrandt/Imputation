# rfr-gapfill

[![CI](https://github.com/inkenbrandt/Imputation/actions/workflows/ci.yml/badge.svg)](https://github.com/inkenbrandt/Imputation/actions/workflows/ci.yml)

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
Random Forest, the operational fill API and the artificial-gap generator are
implemented and usable, as are the validation metrics, the run manifest, the
synthetic site the tests run against and the artificial-gap validation workflow
that ties them together into one call. The reporting layer above it is in too:
gap-length sensitivity tables across arms and sites, the published Table S3
medians as data with an explicit comparison against them, figures drawn from
those tables, bias-IQR uncertainty diagnostics, and site- and
ecosystem-stratified reports across a multi-site study, with the Table S10 Welch
comparison reproduced from the published per-site results. A command-line
interface runs validation and filling in batch over the same API. What remains is
the FLUXNET2015 adapter; FLUXNET files already work through a column map and a few
lines of pandas ([`docs/fluxnet.md`](docs/fluxnet.md)), or through the command line.

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
| Artificial-gap generator and allocation bases | done |
| Validation metrics and energy-balance ratio | done |
| Run manifests and JSON provenance export | done |
| Synthetic site generator and known gaps | done |
| Paper-validation workflow (`validate_rfr`) | done |
| Gap-length sensitivity tables, published benchmarks, plots | done |
| Legacy `fluxlib` compatibility audit and mode (optional Step 20) | done |
| Supplementary uncertainty diagnostics (bias IQR, Table S8 as data) | done |
| Multi-site and ecosystem-stratified reports, Table S10 (Step 20A) | done |
| User documentation, tested quick start (Step 21) | done |
| Command-line interface and configuration files (Step 22) | done |
| Continuous integration on every pull request (Step 23) | done |
| Release checklist for 0.1.0 (Step 24) | done |
| FLUXNET2015 adapter | not started |

## Documentation

| Page | Read it for |
|---|---|
| [`docs/examples.md`](docs/examples.md) | four runnable Jupyter notebooks, from a first fill to the published medians |
| [`docs/method.md`](docs/method.md) | what the method does and why: drivers, receptive limiter, model, ORF |
| [`docs/validation.md`](docs/validation.md) | the artificial-gap experiment, leakage safety, reading and comparing results |
| [`docs/assumptions.md`](docs/assumptions.md) | every point the paper leaves open, the package's conventions, known differences |
| [`docs/fluxnet.md`](docs/fluxnet.md) | preparing FLUXNET2015 files |
| [`docs/supplement_benchmarks.md`](docs/supplement_benchmarks.md) | the published numbers, and where each one comes from |
| [`docs/method_spec.md`](docs/method_spec.md) | the frozen contract the code is tested against |

## Install

Requires Python 3.10+. The package is not on PyPI yet; install it from a clone:

```bash
git clone https://github.com/inkenbrandt/Imputation
cd Imputation

# with uv
uv venv
uv pip install -e ".[dev]"

# or with pip
python -m venv .venv && .venv/Scripts/activate   # .venv/bin/activate on POSIX
pip install -e ".[dev]"
```

`pip install -e .` is enough to use the package; `[dev]` adds the test and lint
tools. Notebooks and `matplotlib` are the optional `notebooks` extra and are
never required by the core. PyYAML, needed only to read YAML configuration files,
is the optional `yaml` extra; JSON configuration files need nothing.

## Quick start

A complete run on a synthetic site, with nothing to download. It validates RFR3
on NEE, H and LE by hiding known stretches of measured data and predicting them,
then fills the genuine gaps in LE. It takes about a minute on a laptop, and the
test suite runs it exactly as written here.

<!-- quickstart:begin -->
```python
from rfrgapfill import FeatureConfig, RFRGapFiller, synthetic_site, validate_rfr

site = synthetic_site()  # one seeded year of half-hourly data at 45 deg N
config = site.config(
    "RFR3",
    features=FeatureConfig(daily_statistic_strategy="rolling_available"),  # A4
    hyperparameter_grid={"n_estimators": (50,)},  # one small point, for speed (A1)
    n_jobs=-1,
)

# 1. Artificial-gap validation: hide known stretches, predict them, score them.
report = validate_rfr(
    site.frame,
    config=config,
    targets=["NEE", "H", "LE"],
    qc_columns=site.qc_columns(),
    gaps=site.known_gaps(),
)
table = report.to_frame()
scores = table[table["subset"] == "all"][["target", "gap_class", "n", "r2", "rmse", "bias"]]
print(scores.round(3).to_string(index=False))

# 2. Real-gap filling: train on the measured rows, fill the genuine gaps.
filler = RFRGapFiller(config).fit(site.frame, target="LE", qc_column="LE_QC")
result = filler.fill(site.frame)
print(result.report.summary())
print(result.frame["LE_fill_method"].value_counts().to_string())
```
<!-- quickstart:end -->

It prints something like this (abridged; the last digits depend on library
versions):

```text
target gap_class    n     r2   rmse   bias
   NEE       all 1309  0.954  1.027 -0.097
   NEE     short  144  0.935  0.925  0.122
   NEE      long  573  0.966  1.179 -0.079
   NEE very_long  592 -0.524  0.881 -0.168
     H       all 1354  0.956 19.410 -0.742
   ...
LE: filled 849 of 849 candidate row(s) with RFR3.
LE_fill_method
observed      15130
pre_filled     1541
RFR3            849
```

What each piece is doing:

- `site.config("RFR3", ...)` fills in the site's cadence, latitude, column
  mapping and seed. With your own data you build an `RFRConfig` directly.
- `daily_statistic_strategy="rolling_available"` matters: under the default, a
  gap covering whole days has no daily target statistics, so 7-day and 30-day
  gaps get no predictions at all ([A4](docs/assumptions.md#a4-days-with-too-few-measured-values)).
- The one-point grid is for speed. Drop `hyperparameter_grid` to search the
  package's default grid.
- `site.known_gaps()` is a fixed plan of one 30-day, two 7-day and three 24-hour
  gaps. Leave out `gaps=` to draw the paper's scenario instead (about 25% of the
  measured data, mixed 20/30/50).
- NEE's 30-day R2 is negative while its RMSE is the smallest of any NEE class.
  That gap falls in winter, when NEE hardly varies, and R2 compares the error
  with that tiny variance. Read the metrics together
  ([`docs/validation.md`](docs/validation.md#read-the-metrics-together)).
- `result.frame` is a new frame: the input is never modified. `pre_filled` rows
  were already gap-filled before they reached the package and are left alone.

`python examples/synthetic_example.py` goes further: both configurations, per
gap class and day/night subset, the energy balance, and the published medians
printed for reference. The [example notebooks](docs/examples.md) cover the same
ground step by step, with figures, and go on to a FLUXNET-format file and the
command line.

## Expected input

One site at a time, as a pandas `DataFrame` with one row per time step:

- **Time.** A `DatetimeIndex`, or a timestamp column named with `timestamp=`.
  Numbers are refused as timestamps, because pandas would read them as
  nanoseconds since 1970. Rows are sorted for you; duplicate timestamps raise
  unless you choose how to resolve them. Missing rows are fine.
- **Cadence.** Half-hourly is the paper's; other regular cadences such as hourly
  work. Declare it with `frequency=` or let it be inferred.
- **Targets.** One continuous flux column per model. To compare with the paper,
  name them `NEE` (µmol m-2 s-1), `H` and `LE` (W m-2).
- **A QC column per target**, strongly recommended. Values in
  `observed_qc_values` (default `0`, FLUXNET's "measured") are measurements;
  anything else, including a missing flag, is treated as already gap-filled.
- **Drivers** for the chosen mode, in the units of the table below. The package
  neither fills nor converts them.
- **Latitude or hemisphere**, for the season feature. One is required.
- **`NETRAD` and `G`** as well, if you want the energy-balance check.
- **Missing values as `NaN`.** A sentinel such as FLUXNET's `-9999` would be read
  as a real number.
## Specification first

The implementation is checked against a frozen specification rather than against
prose in the paper:

- [`docs/method_spec.md`](docs/method_spec.md) — the contract: driver lists,
  receptive-limiter features, gap scenario, metrics, and a **Known ambiguities**
  table (A1–A12) covering every point where the paper is silent.
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
  MDS-equivalent drivers.
- **RFR10** — the same, with seven more drivers.

| Driver | Canonical name | FLUXNET2015 column | Units | RFR3 | RFR10 |
|---|---|---|---|:---:|:---:|
| Downward shortwave radiation | `shortwave` | `SW_IN_F` | W m-2 | ✓ | ✓ |
| Vapour pressure deficit | `vpd` | `VPD_F_MDS` | hPa | ✓ | ✓ |
| Air temperature | `air_temperature` | `TA_F_MDS` | degC | ✓ | ✓ |
| Net radiation | `net_radiation` | `NETRAD` | W m-2 | | ✓ |
| Wind speed | `wind_speed` | `WS` | m s-1 | | ✓ |
| Wind direction | `wind_direction` | `WD` | degrees | | ✓ |
| Soil heat flux | `soil_heat_flux` | `G_F_MDS` | W m-2 | | ✓ |
| Soil temperature | `soil_temperature` | `TS_F_MDS` | degC | | ✓ |
| Relative humidity | `relative_humidity` | `RH` | % | | ✓ |
| Soil water content | `soil_water_content` | `SWC_F_MDS` | % | | ✓ |

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

### Historical `fluxlib` compatibility

[`docs/fluxlib_audit.md`](docs/fluxlib_audit.md) audits the paper-era `fluxlib`
code the article cites. That code computes the daily statistics *before* the
artificial gaps are hidden, so held-out truth reaches its own predictors, and it
departs from the article in several other places. `feature_mode="legacy_fluxlib"`
reproduces its derivation so you can measure how much that flatters a score: it
warns on every validation build, labels its arms `RFR3-legacy` / `RFR10-legacy`,
and is never paper faithful. `hyperparameter_preset="legacy_fluxlib"` (the
archived `GridSearchCV` grid) and `"legacy_fluxlib_fixed"` (the parameters its
pipelines actually fitted) are the matching model settings. No default follows the
old code.

```python
historical = config.replace(
    features=FeatureConfig(feature_mode="legacy_fluxlib"), cv_folds=3
).with_hyperparameter_preset("legacy_fluxlib_fixed")
```

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

## Artificial gaps

`GapScenarioGenerator` builds the paper's validation scenario: roughly 25% of the
genuinely observed values withheld as 24-hour, 7-day and 30-day gaps mixed
20/30/50, each interval needing at least 50% real measurement underneath it.

```python
from rfrgapfill import GapScenarioGenerator

manifest = GapScenarioGenerator(config).generate(df, target="LE", qc_column="LE_QC")

manifest.summary()      # exactly how the mixture was constructed
manifest.to_frame()     # one row per gap: id, class, start, end, duration, coverage
holdout = manifest.mask(df.index)   # step 1 of the leakage-safe workflow
```

Pass several targets to reproduce the paper's joint NEE/H/LE design — one set of
gap locations, scored for all three:

```python
manifest = GapScenarioGenerator(config).generate(
    df, target=["NEE", "H", "LE"], qc_column={"NEE": "NEE_QC", "H": "H_QC", "LE": "LE_QC"}
)
```

### The 20/30/50 ambiguity is yours to choose (A3)

The paper does not say whether those percentages count gap *events* or withheld
*half-hours*, and the two readings give very different designs — a 20/30/50 mix
of events puts over 80% of the withheld records in the 30-day class. Both are
implemented, the default is `missing_records`, and every manifest reports the
achieved mix on **both** bases next to what was requested:

```python
config = RFRConfig(
    mode="RFR10",
    hemisphere="north",
    validation=ValidationConfig(gaps=GapScenarioConfig(allocation_basis="gap_events")),
)
```

Nothing is quietly corrected to look exact. What a scenario really withholds is
measured from the placed intervals, and an achieved fraction or mix outside the
configured tolerance, a class the series is too short to hold, or a design that
could not be placed at all is a `GapScenarioWarning` or a `GapError` — never a
silently thinner set of gaps.

## A site to try it on

Nothing in this package's tests downloads anything. `synthetic_site()` builds a
deterministic year of half-hourly data with structure you can predict: a
clear-sky diurnal radiation cycle from the solar declination, a seasonal
temperature swing with an afternoon thermal lag, VPD derived from temperature and
humidity, a soil temperature damped and delayed behind the air, and NEE, H and LE
that are real functions of all of it.

```python
from rfrgapfill import synthetic_site

site = synthetic_site()             # 17520 rows, 2018, 45 deg N, FLUXNET column names
site.frame                          # ten drivers, NEE/H/LE and their QC flags
site.truth                          # the same fluxes before measurement noise
config = site.config("RFR10")       # cadence, latitude, column map and seed filled in
```

The drivers are complete, as the paper's pre-filled meteorology is; only the
fluxes carry gaps. Roughly 5% of rows are bursty instrument outages shared by all
three targets and 10% are flagged as gap-filled before ingestion — NEE's flagged
blocks start at night, the way friction-velocity filtering removes calm nights.
`noise_scale=0.0`, `real_gap_fraction=0.0` and `pre_filled_fraction=0.0` strip
each of those away when a test needs an exact relationship instead.

Two things about it are exact rather than approximate. Sensible heat is the
residual of available energy after latent heat, so

```python
(site.truth.H + site.truth.LE).sum() / (site.frame.NETRAD - site.frame.G_F_MDS).sum()
# 0.85 == site.energy_balance_closure
```

and a negative `latitude` flips the seasons, the phenology and the temperature
cycle together, because all three are read off the same solar geometry rather
than off the calendar month.

### Known gaps

`site.known_gaps()` returns an ordinary `GapManifest` for a fixed plan — one
30-day gap, two 7-day gaps and three 24-hour gaps at documented offsets — so a
test can state its row counts in advance:

```python
manifest = site.known_gaps()        # shared across NEE, H and LE
manifest.to_frame()                 # 6 rows, durations of exactly 1, 7 and 30 days
holdout = manifest.mask(site.frame.index)
```

That is a hand-checkable fixture, not the paper's scenario: for 25% withheld at
20/30/50, run `GapScenarioGenerator` against `site.frame` like any other input.
Doing so is also the quickest way to see ambiguity A7 at work — with real outages
underneath, a 25% request lands nearer 18%, and the manifest says so rather than
adjusting itself.

## Scoring a fill

`rfrgapfill.metrics` is the paper's evaluation quantities and nothing else — R2,
the regression slope, RMSE, its bias definition, and the energy-balance ratio.
Each is an independent function of the values it is handed, so the same call
scores a whole run, one gap class or one daytime subset:

```python
from rfrgapfill import core_metrics, bias, compare_energy_balance

scored = core_metrics(measured, predicted)
scored.r2, scored.slope, scored.rmse, scored.bias
scored.n, scored.n_offered          # pairs scored, rows offered
scored.to_dict()                    # straight into the run manifest

bias(measured, predicted)           # (sum(predicted) - sum(measured)) / n
```

Two rules apply before anything is computed. Series carrying different indexes
are **refused**, not aligned — silent realignment is how a prediction gets scored
against the wrong half hour. Incomplete pairs are dropped from both sides at
once, so a row the model left unfilled for want of a predictor leaves the score
entirely rather than poisoning it; `n` is what survived, and it is the `n` in the
bias denominator.

An undefined metric is `None`, never `NaN` or zero. An empty subset has no RMSE,
constant measurements have no slope, and zero available energy has no EBR —
`None` reaches a JSON manifest as `null` and cannot propagate silently through a
later mean.

### Which R2 (A12)

The paper prints R2 next to a slope without saying which quantity it is. The
default is `residual` — `1 - SS_res/SS_tot`, scikit-learn's `r2_score` — because
the alternative, the squared Pearson correlation, is invariant to any affine
rescaling of the predictions and so scores a systematically offset series as
perfectly as an unbiased one. A package that reports bias in the next column
should not report an R2 that cannot see it. Both are implemented, and the choice
is recorded in every result:

```python
ValidationConfig(r2_definition="squared_correlation")
```

The two coincide when predictions are unbiased and shrunk toward the mean, which
is where Supplementary Table S3 sits — its median R2 and median slope agree to
about 0.01 in every RFR row — so neither reading is paper exact.

### Energy balance

```python
comparison = compare_energy_balance(
    measured_sensible_heat=h_measured,
    measured_latent_heat=le_measured,
    filled_sensible_heat=h_filled,
    filled_latent_heat=le_filled,
    net_radiation=df["NETRAD"],
    soil_heat_flux=df["G"],
)
comparison.measured, comparison.filled, comparison.difference
```

`sum(H + LE) / sum(NETRAD - G)` over the artificial-gap rows, measured against
filled. A row is used only where every component it needs is present, and both
ratios share one row set — otherwise the difference would report the change in
interval as much as the change in flux.

## Run manifests

A filled column is only as trustworthy as the record of what produced it.
`RunManifest` assembles the description every stage already writes — the
configuration, the column mapping, the QC rule, the grid and the parameters it
chose, the time axis, the row accounting — into one JSON document:

```python
from rfrgapfill import RunManifest

manifest = RunManifest.from_fill(filler, result)
manifest.save("LE_run.json")        # checks completeness, then writes
manifest.settings_digest            # sha256 of the settings, and only the settings
```

and, for an artificial-gap run, the placed scenario alongside them:

```python
manifest = RunManifest.from_validation(
    config=config, model=model, gaps=gap_manifest, features=feature_set
)
```

Three things make it more than a dictionary dump.

**The required list is checked.** `REQUIRED_FIELDS` maps every item the
specification asks a run to record onto its path in the document, and
`require_complete()` names any the manifest cannot supply. A gap manifest is
required of a validation run and not of a fill, which places no intervals; a
hemisphere only where the season feature is built, since the ORF benchmark builds
none. `save()` runs the check before writing — the file is what outlives the
session that could have explained it.

**The resolved ambiguities travel with the result.** A1–A12 are not merely
exposed in configuration; the choice this run made for each is written into every
manifest, under a note stating plainly that none of them reproduces the paper
exactly:

```python
manifest.to_dict()["ambiguities"]["choices"]["A3"]["settings"]
# {'allocation_basis': 'missing_records'}
```

**The digest covers the design, not the run.** `settings_digest` is a SHA-256
over the environment versions, the target and the whole validated configuration —
and over nothing that varies between two runs of the same design, so the
timestamp, the row counts, the placed intervals and the scores are all excluded.
Two manifests sharing a digest were configured identically.

Reading one back is `load_manifest(path)`, which returns the document as data
rather than a live configuration: an archived manifest records a run that already
happened. To rerun the design it records, `RFRConfig.from_dict(document["config"])`
rebuilds the configuration through the constructors, which revalidate every
setting rather than trusting a hand-editable file. `to_dict()` is a plain mapping,
so `yaml.safe_dump(manifest.to_dict())` works with any YAML library you already
have; the package takes no dependency on one.

## Artificial-gap validation

`validate_rfr` is the paper's experiment in one call: place the gaps, hide the
truth, build leakage-safe features, fit on what remains, predict the withheld
intervals and score them.

```python
from rfrgapfill import RFRConfig, validate_rfr

report = validate_rfr(
    df,
    config=RFRConfig(mode="RFR10", frequency="30min", latitude=45.0, column_map=columns),
    targets=["NEE", "H", "LE"],
    qc_columns={"NEE": "NEE_QC", "H": "H_QC", "LE": "LE_QC"},
)

report.to_frame()              # target x gap class x subset: R2, slope, RMSE, bias, n
report["LE"].overall.r2        # one number
report["LE"].metric(subset="nighttime", gap_class="very_long")
report.gaps.summary()          # what the scenario asked for and what it achieved
report.energy_balance          # measured vs filled EBR, over the same rows
report.manifest("LE").save("LE_validation.json")
```

The withheld observations **are** the test set (~25%) and the remaining eligible
observations are the training set — Figure 2 of the paper. There is no random
row-wise split anywhere in the workflow, because that would destroy the temporal
gap structure the method is about.

Every target of a run shares **one** set of gap locations, which is the paper's
joint NEE/H/LE rule and what makes the arms comparable; pass `gaps=` to score a
second arm, or an ORF benchmark, on exactly the same intervals:

```python
rfr = validate_rfr(df, config=config, targets=targets, qc_columns=qc)
orf = validate_rfr(df, config=config.as_orf(), targets=targets, qc_columns=qc, gaps=rfr.gaps)
```

Metrics come back for `all`, `daytime` and `nighttime` (`SW_IN > 20 W m-2`) and
again per gap class, together with the spread of bias across the individual gaps
of each class. A row whose radiation is missing belongs to neither day nor night:
the split is defined by a measurement, and putting an unknown row on one side of
it would silently invent that measurement.

**Long gaps need a deliberate choice (A4).** Under the documented default
`daily_statistic_strategy="missing"`, a gap covering a whole calendar day has no
daily target statistics, so no row of it has a complete feature vector and none
is predicted or scored. That is the honest reading of the paper's silence, not a
defect — and it does not pass quietly: the run raises a `ValidationWarning` and
the row accounting shows `holdout_rows_with_complete_features = 0`. A 7-day or
30-day run therefore selects a reaching strategy and records it:

```python
config = config.replace(
    features=config.features.replace(daily_statistic_strategy="rolling_available")
)
```

`examples/synthetic_example.py` is the whole thing end to end: both published
configurations over NEE, H and LE on one synthetic year, printing the metric
tables, the gap manifest and the energy balance, and checking that no observed
value changed.

## Comparing by gap length

Scoring one run is `validate_rfr`; comparing runs is `gap_length_table`. It takes
results - or tidy tables read back from disk, which is what a multi-site
reproduction actually has - and lines them up:

```python
from rfrgapfill import gap_length_table, gap_length_pivot, median_across_sites

table = gap_length_table({"US-Ha1": [rfr3, rfr10], "FI-Hyy": [rfr3_hyy, rfr10_hyy]})
# site | target | method | mode | gap_class | subset | units | n | n_offered | r2 | slope | rmse | bias

medians = median_across_sites(table)
gap_length_pivot(medians, metric="r2", gap_class="very_long")
#         RFR3  RFR10
# NEE     0.74   0.74
# H       0.76   0.89
# LE      0.74   0.83
```

Gap classes come back in **duration order** - `short`, `long`, `very_long`, with
the row pooling all of them labelled `all` - because alphabetical order would put
`long` before `short` and make every sensitivity plot read backwards. Nothing in
this layer computes a metric: every number came out of `rfrgapfill.metrics`
inside a validation run, so a table can be wrong about labelling but never about
values. Two rows landing in one cell is an error rather than a silent average,
and a site that contributed the same cell twice is refused rather than counted
twice in a median.

### The published medians

Supplementary Table S3 is carried as data, every value traceable to
[`docs/supplement_benchmarks.md`](docs/supplement_benchmarks.md) — the test suite
re-reads that document and checks the constants against it, so the two cannot
drift apart:

```python
from rfrgapfill import benchmark_table, compare_to_benchmarks, convert_nee_to_carbon_units

gap_length_pivot(benchmark_table(), metric="r2", gap_class="all")
#         MDS   RFR3  RFR10
# NEE    0.72   0.78   0.84
# H      0.67   0.78   0.90
# LE     0.63   0.75   0.85

compare_to_benchmarks(medians)   # run | published | difference | comparable | note
```

These are **reproduction benchmarks** for a run over matching FLUXNET inputs with
matching preprocessing — medians across the paper's 94-site subset, not pass/fail
thresholds for any station. Nothing in that module asserts anything, and every
comparison row says which population the published value came from.

**Units are refused, not guessed (A10).** Table S3 reports NEE `RMSE` and `bias`
in `g C m-2 d-1`; this package models NEE in `umol m-2 s-1`, and the paper does
not say how it aggregated half-hourly residuals into daily carbon. So those two
cells come back `comparable=False` with a note, until you convert explicitly:

```python
compare_to_benchmarks(convert_nee_to_carbon_units(medians))
```

`R2` and `slope` are dimensionless and compare from the start. The conversion is
a rate conversion — `12.011e-6 × 86400` — and is documented as exactly that, not
as a reconstruction of the paper's aggregation.

### Figures

`rfrgapfill.plotting` draws those tables and nothing else, so a figure cannot
disagree with the table printed beside it:

```python
from rfrgapfill import plot_gap_length_grid
plot_gap_length_grid(medians, metric="r2")     # one panel per flux, shared y axis
```

`matplotlib` is an optional dependency (`pip install "rfr-gapfill[notebooks]"`),
imported only when a figure is drawn and named in the error when it is missing.

## Command line

Batch runs go through `rfr-gapfill`, a thin shell over the same API. The
configuration file becomes an `RFRConfig` through `load_config`, and the two
subcommands call `validate_rfr` and `RFRGapFiller` exactly as a script would, so a
command-line run and a notebook run of one configuration produce the same numbers:

```bash
rfr-gapfill validate site.csv --target LE --mode RFR10 --config run.json --output validation/
rfr-gapfill fill site.csv --target LE --mode RFR10 --config run.json \
  --qc-column LE_QC --output filled.csv
```

The configuration file is the mapping `RFRConfig.to_dict()` writes, as JSON, or as
YAML with the `yaml` extra. Only `mode` and what the constructor demands anyway
are required; every other setting takes its documented default, and a misspelt
key is refused rather than ignored:

```json
{
  "mode": "RFR3",
  "latitude": 45.0,
  "site_id": "XX-Xxx",
  "frequency": "30min",
  "features": {"daily_statistic_strategy": "rolling_available"},
  "column_map": {
    "shortwave": "SW_IN_F", "vpd": "VPD_F_MDS", "air_temperature": "TA_F_MDS",
    "timestamp": "TIMESTAMP"
  }
}
```

The `config` section of any run manifest is itself a valid configuration file, so
a recorded design reruns as it stands, and `--mode` overrides the file's mode so
one file serves both arms.

| Subcommand | Writes |
|---|---|
| `validate` | into `--output`: `metrics.csv`, `bias_spread.csv`, `energy_balance.csv`, `gaps.csv`, `predictions.csv`, `summary.txt`, and `<method>_<target>_manifest.json` per target |
| `fill` | `--output` (the input columns plus the six provenance columns) and `<stem>.manifest.json` beside it |

The command line owns only what the library leaves to its caller: reading the CSV
(`--na-value -9999` for FLUXNET's missing marker), finding the timestamp column
(`--timestamp`, or `column_map.timestamp`; `--timestamp-format %Y%m%d%H%M` for an
integer `TIMESTAMP_START`), and writing results. Every manifest it writes records
the command line and the SHA-256 of the input and configuration files. Existing
output is never replaced without `--overwrite`, and a run that cannot proceed
exits with status 1 and a one-line message. `rfr-gapfill <command> --help` lists
every option.

## Known differences from the paper

The article leaves twelve implementation points open. Each has a default, a
setting, and a record in every run manifest; none of them may be called paper
exact. [`docs/assumptions.md`](docs/assumptions.md) explains each one and how to
change it.

| ID | Open point | Default here |
|---|---|---|
| A1 | the `GridSearchCV` grid | a documented 8-candidate grid; `fluxlib`'s grids as named presets |
| A2 | radiation categories at exactly 10 and 100 W m-2 | `< 10` weak, `10 to 100` medium, `> 100` strong |
| A3 | whether 20/30/50 counts gap events or withheld records | withheld records; the achieved mix is reported on both bases |
| A4 | daily statistics for a day with no visible measurements | left missing, so whole-day gaps need a reaching strategy |
| A5 | cross-validation inside the grid search | 5 unshuffled folds; time-series folds as an enhancement |
| A6 | whether the original code was leakage safe | it was not; the leakage-safe mode is the default |
| A7 | hitting exactly 25% withheld | the achieved fraction is reported against a tolerance |
| A8 | the Table S8 normalized uncertainty ratio | bias IQR reported; the ratio is not computed |
| A9 | hemisphere at the equator | `latitude >= 0` is north; `hemisphere=` overrides |
| A10 | units of published NEE RMSE and bias | model units; published cells in other units are not compared |
| A11 | daily standard deviation and quantile conventions | sample standard deviation, linear quantiles |
| A12 | which R2 | `1 - SS_res/SS_tot`; squared correlation available |

Beyond those twelve: MDS is not implemented (it appears only as published
numbers), sites are never pooled into one model, and the supplement disagrees
with itself in a few places the package records rather than corrects
([`docs/assumptions.md`](docs/assumptions.md#where-the-supplement-disagrees-with-itself)).

## Development

```bash
pytest              # tests
pytest -m "not slow and not supplement"   # the fast subset
pytest --cov        # tests with branch coverage; fails below 90%
pytest -m notebooks # execute the example notebooks (needs the notebooks extra)
ruff check .        # lint
ruff format .       # format
mypy                # type check (strict, src/rfrgapfill)
```

Test markers: `slow`, `fluxnet` (needs real FLUXNET2015 input), `supplement`
(needs the journal supplementary files). None of that data is committed here.

Every pull request and every push to `main` runs
[`.github/workflows/ci.yml`](.github/workflows/ci.yml):

| Job | What it checks |
|---|---|
| Lint and types | `ruff check`, `ruff format --check`, strict `mypy` |
| Tests | the full suite with branch coverage, on Python 3.10 to 3.14 and on Windows; the `yaml` extra and matplotlib are installed so their tests run |
| Dependency floors | the full suite on Python 3.10 at the lowest version of every dependency `pyproject.toml` allows, so a declared minimum cannot quietly become false |
| Example notebooks | executes every notebook in `examples/notebooks/` with the `notebooks` extra installed |
| Package | builds the sdist and the wheel from it, runs `twine check`, installs the wheel into a clean environment and runs the README quick start and the console script against it |

The `supplement` and `fluxnet` tests skip in CI because their data is not in the
repository. Run them locally before changing anything they cover.

## Citing

Cite the paper for the method and [`CITATION.cff`](CITATION.cff) for this
implementation. This package is not affiliated with the authors of Zhu et al.
(2022).

## License

MIT — see [`LICENSE`](LICENSE).
