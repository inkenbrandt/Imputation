# The RFR method, as this package implements it

This page is for someone who wants to use the package and understand what it
does. The contract the code is tested against is
[`method_spec.md`](method_spec.md); where this page and the specification
disagree, the specification is right and this page is a bug.

Companion pages: [`validation.md`](validation.md) (the artificial-gap experiment
and how to read it), [`assumptions.md`](assumptions.md) (every choice the paper
leaves open), [`fluxnet.md`](fluxnet.md) (FLUXNET2015 input), and
[`supplement_benchmarks.md`](supplement_benchmarks.md) (the published numbers).
To see the method run before reading about it, open
[`01_getting_started.ipynb`](../examples/notebooks/01_getting_started.ipynb), one
of the [example notebooks](examples.md).

---

## What the method is for

An eddy-covariance tower measures the exchange of CO2 (net ecosystem exchange,
`NEE`), sensible heat (`H`) and latent heat (`LE`) between an ecosystem and the
atmosphere, usually every half hour. Instrument faults, power cuts, bad weather
and quality filtering leave gaps, and annual budgets need a complete series, so
the gaps are filled with a model driven by meteorology.

The long-standing standard is marginal distribution sampling (MDS), which fills a
gap from measurements taken under similar conditions nearby in time. Zhu et al.
(2022) show that its skill falls as gaps lengthen, and propose **Random Forest
Robust (RFR)**: a random forest regressor fed the meteorological drivers plus a
small set of engineered features the paper calls the *receptive limiter*. They
validate it with artificial gaps of 24 hours, 7 days and 30 days at 94 FLUXNET2015
sites for all three fluxes and at 194 sites for NEE with the three MDS-equivalent
drivers.

The published medians make the motivation concrete. For 30-day gaps, the median
R2 of H across the 94 sites is 0.62 for MDS, 0.76 for RFR3 and 0.89 for RFR10
(Supplementary Table S3, [`supplement_benchmarks.md`](supplement_benchmarks.md)
section 3). Those are medians across sites; individual sites vary widely, and
nothing in this package expects a given site to reach them.

## The workflow at a glance

Everything is fitted **per site and per target**. No model ever sees two sites'
data, and NEE, H and LE each get their own features and their own forest.

1. **Put the data on a time axis.** Timestamps are validated, sorted and checked
   for duplicates, and the cadence is inferred or declared
   (`rfrgapfill.time`). Every duration afterwards is elapsed time, never a row
   count.
2. **Decide which target values are genuine measurements.** A QC flag separates
   measured values from values that were already gap-filled before they reached
   you. Only measured values train the model or feed its daily statistics.
3. **Build the features:** the drivers of the chosen mode, plus the receptive
   limiter (`rfrgapfill.features`).
4. **Fit a random forest** with a grid search over its hyperparameters
   (`rfrgapfill.model`).
5. **Predict only where the target is missing.** A present value is never
   replaced (`rfrgapfill.fill`).
6. **Record what was done:** configuration, columns, QC rule, grid, chosen
   parameters, row accounting and every ambiguity choice go into a JSON run
   manifest (`rfrgapfill.provenance`).

Validation runs the same steps with one change: part of the *measured* data is
hidden first and scored afterwards. That is [`validation.md`](validation.md).

## Drivers: RFR3 and RFR10

The two named configurations differ only in their drivers. RFR3 uses the three
drivers MDS uses; RFR10 adds seven more.

| Mode | Canonical name | FLUXNET2015 column | Units |
|---|---|---|---|
| RFR3, RFR10 | `shortwave` | `SW_IN_F` | W m-2 |
| RFR3, RFR10 | `vpd` | `VPD_F_MDS` | hPa |
| RFR3, RFR10 | `air_temperature` | `TA_F_MDS` | degC |
| RFR10 | `net_radiation` | `NETRAD` | W m-2 |
| RFR10 | `wind_speed` | `WS` | m s-1 |
| RFR10 | `wind_direction` | `WD` | degrees |
| RFR10 | `soil_heat_flux` | `G_F_MDS` | W m-2 |
| RFR10 | `soil_temperature` | `TS_F_MDS` | degC |
| RFR10 | `relative_humidity` | `RH` | % |
| RFR10 | `soil_water_content` | `SWC_F_MDS` | % |

Canonical names are what the code uses internally. Your columns can be called
anything; a `ColumnMap` says which of your columns is which driver, and
`ColumnMap.fluxnet2015("RFR10")` is the reference FLUXNET mapping. Units are
recorded, not converted: supply the drivers in the units above.

Drivers are expected to be complete, as the paper's pre-filled FLUXNET
meteorology is. The package never fills a driver itself. A row missing any
driver is not used for training and is not predicted; it stays missing and is
counted.

`net_radiation` and `soil_heat_flux` are also needed for the energy-balance ratio
of a validation run, whichever mode you choose.

## The receptive limiter

These four groups of features are what make RFR more than a random forest over
meteorology. Switching them off gives the paper's ORF benchmark (below).

| Feature | Built from | Default | Setting |
|---|---|---|---|
| Radiation category: `weak` / `medium` / `strong` | `shortwave` | `< 10`, `10 to 100`, `> 100` W m-2 | `FeatureConfig.radiation_thresholds`, `boundary_convention` |
| `time_distance_hours` | the timestamps | hours since the first timestamp of the series | none; fixed at fit time for later fills |
| Season: `winter` / `spring` / `summer` / `autumn` | calendar month and hemisphere | Dec-Feb is winter in the north, summer in the south | `RFRConfig.hemisphere` or `latitude` (one is required) |
| `<target>_daily_q1`, `_q2`, `_q3`, `_std` | the target itself | per calendar day, from measured values only | `FeatureConfig.daily_statistic_strategy`, `min_daily_observations`, `daily_std_ddof` |

`feature_names(config, target="LE")` lists the full feature order a model is
fitted on, and `receptive_limiter_features(config, target="LE")` lists exactly
what the limiter adds.

### Why the daily statistics need care

The daily quartiles and standard deviation are computed **from the flux being
predicted**. In ordinary filling that is harmless: a missing value cannot
contribute to its own day's statistics. In validation it is the central risk. If
the statistics are computed before the artificial gaps are hidden, the hidden
values help build the features that are then used to predict them, and the
score is flattered. The package's default `feature_mode="paper_safe"` hides the
gaps first and computes the statistics only from what the model is allowed to
see; [`validation.md`](validation.md) explains how that is enforced and checked.

The same features raise a second question the paper does not answer: what should
a day with no measured values get? Under the default, nothing, which means a
gap covering whole days cannot be predicted. That is ambiguity A4, and it matters
for every gap longer than a day; see [`assumptions.md`](assumptions.md#a4-days-with-too-few-measured-values).

## The model

`RFRModel` is scikit-learn's `RandomForestRegressor` inside `GridSearchCV`:

- the grid is `DEFAULT_HYPERPARAMETER_GRID` unless you pass
  `hyperparameter_grid=` or a named `hyperparameter_preset=`. The paper says a
  grid search was used but not what it searched, so no grid is "paper exact"
  (A1);
- the forest, the folds and the gap sampler all take their seed from
  `RFRConfig.random_state`, so a run is reproducible;
- a row missing any predictor is dropped at fit and left `NaN` at prediction,
  rather than letting a recent scikit-learn forest impute it silently.
  `model.fit_report` says how many rows each feature cost;
- `n_jobs` is applied to the forest, not to the grid search, so one setting cannot
  multiply into folds x candidates x trees processes;
- `model.save(path)` and `RFRModel.load(path)` round-trip the model with its
  configuration, feature order and environment versions, and the configuration
  is validated again as it loads.

## The ORF benchmark

The paper measures what the receptive limiter contributes by comparing RFR with
an ordinary random forest (ORF) over the same drivers (Supplementary Figure S1).
`config.as_orf()` derives that arm, changing nothing but the limiter switch, and
`require_orf_pairing(rfr_config, orf_config)` refuses a pair that differs in
anything else. Score both on the same gaps (`validate_rfr(..., gaps=...)`) and
the difference is the limiter's effect. The ORF arms are labelled `ORF3` and
`ORF10`, and a configuration with the limiter off is never reported as paper
faithful.

## Filling real gaps

`RFRGapFiller` fits on one site's measured values of one target and fills that
target's gaps. The rules it enforces:

- only measured, QC-accepted values train the model;
- a present value is never replaced. Values that arrived already gap-filled are
  carried through and labelled `pre_filled`; `fill(..., refill_pre_filled=True)`
  replaces them deliberately;
- a row lacking a predictor stays missing and is labelled
  `unfilled_incomplete_features`;
- your DataFrame is never modified. The result is a new frame with six
  provenance columns per target: `_original`, `_filled`, `_is_observed`,
  `_is_filled`, `_fill_method` and `_model_version`.

The [README](../README.md#filling-real-gaps) shows the call and the columns.

## What the package deliberately does not do

- **MDS is not implemented.** It appears only as published numbers to compare
  against.
- **Sites are never pooled** into one model. Multi-site work compares per-site
  *scores* (`rfrgapfill.sites`).
- **Drivers are never gap-filled, and units are never converted.** Both are the
  caller's responsibility and both are recorded.
- **No threshold decides whether a site "passes".** Skill varies by ecosystem
  and gap length, and the published numbers are medians across sites, not
  targets for one.
- **The supplement's normalized uncertainty ratios (Table S8) are not
  computed**, because their denominator has not been reconstructed (A8).

## Where each piece lives

| Module | Responsibility |
|---|---|
| `rfrgapfill.schema`, `rfrgapfill.config` | canonical names, `ColumnMap`, `RFRConfig` and every validated setting |
| `rfrgapfill.time` | timestamps, cadence, elapsed hours |
| `rfrgapfill.features` | drivers and receptive-limiter features |
| `rfrgapfill.leakage` | the leakage-safe feature workflow for validation |
| `rfrgapfill.model` | `RFRModel`: forest, grid search, persistence |
| `rfrgapfill.fill` | `RFRGapFiller`: operational filling and its provenance columns |
| `rfrgapfill.gaps` | artificial-gap scenarios and their manifests |
| `rfrgapfill.validation` | `validate_rfr`: the whole artificial-gap experiment |
| `rfrgapfill.metrics` | R2, slope, RMSE, bias, energy-balance ratio |
| `rfrgapfill.provenance` | `RunManifest`: the JSON record of a run |
| `rfrgapfill.sensitivity`, `rfrgapfill.plotting` | tables and figures across runs, sites and gap lengths |
| `rfrgapfill.benchmarks`, `rfrgapfill.uncertainty` | the published Table S3 medians and Table S8 ranges, as data |
| `rfrgapfill.sites` | site metadata, ecosystem-stratified reports, the Table S10 Welch test |
| `rfrgapfill.synthetic` | a deterministic synthetic site to try everything on |
| `rfrgapfill.legacy` | the historical `fluxlib` feature derivation, for comparison only |
| `rfrgapfill.fluxnet` | the optional FLUXNET2015 adapter: reading, availability and QC reports ([`fluxnet_adapter.md`](fluxnet_adapter.md)) |
| `rfrgapfill.cli` | `rfr-gapfill validate` and `rfr-gapfill fill`, over the same API |
