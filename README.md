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
- [`docs/fluxnet_adapter.md`](docs/fluxnet_adapter.md) — what the optional
  FLUXNET2015 adapter assumes about that data product, what its QC flags mean,
  and the four naming choices (`F1`–`F4`) it makes on your behalf.

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
