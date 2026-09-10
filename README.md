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
Random Forest, the operational fill API and the artificial-gap generator are
implemented and usable, as are the validation metrics, the run manifest, the
synthetic site the tests run against and the artificial-gap validation workflow
that ties them together into one call. The reporting layer above it is in too:
gap-length sensitivity tables across arms and sites, the published Table S3
medians as data with an explicit comparison against them, and figures drawn from
those tables. What remains is the FLUXNET2015 adapter, the supplementary
uncertainty diagnostics and the CLI.

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
| FLUXNET2015 adapter, supplementary uncertainty diagnostics, CLI | not started |

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
happened, and rebuilding a configuration from a hand-editable file would invite
treating it as a validated one. `to_dict()` is a plain mapping, so
`yaml.safe_dump(manifest.to_dict())` works with any YAML library you already
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
