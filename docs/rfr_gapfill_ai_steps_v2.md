# AI Implementation Steps: Build a Python Package for Zhu et al. (2022) RFR Gap Filling

These steps are written for an AI coding agent. Work through them in order. Keep each step small enough to review and test independently.

**Primary rule:** do not silently invent details that are absent from Zhu et al. (2022). Preserve unresolved choices as configuration and document them.

## Progress

| Steps | Status | Modules |
|---|---|---|
| 1-4 | done | `docs/method_spec.*`, `schema.py`, `config.py`, `time.py` |
| 5, 5A, 6 | done | `features.py`, `tests/test_features.py` |
| 7, 8 | done | `model.py`, `fill.py`, `provenance.py` |
| 9, 10 | done | `gaps.py`, `tests/test_gaps.py` |
| 11, 12 | done | `metrics.py` |
| 13 | done | `validation.py` — `validate_rfr()`, `compare_receptive_limiter()` |
| 19 (EBR) | done | reported by `validate_rfr` for H and LE |
| 14-18, 20-24 | outstanding | FLUXNET adapter, reproduction benchmarks, multi-site, CLI, CI, release |

Step 6 surfaced a point the paper does not settle, recorded as **A4a** in
[`method_spec.md`](method_spec.md): a strictly leakage-safe reading makes gaps
longer than one day unpredictable, which the paper's own 7- and 30-day results
rule out. The default is a documented leakage-safe fallback
(`daily_statistics_strategy="nearest_visible_day"`); the strict reading remains
available and its cost is measurable.

---

## Step 1 — Freeze the scientific specification

### Goal

Translate the paper into a short machine-readable implementation specification before writing model code.

### Tasks

- Record the target methods:
  - `RFR3`
  - `RFR10`
- Record the canonical driver list for each.
- Record receptive-limiter features:
  - radiation category;
  - elapsed hours;
  - hemisphere-aware season;
  - daily target Q1, Q2, Q3, and standard deviation.
- Record validation gap durations:
  - 24 h;
  - 7 d;
  - 30 d.
- Record artificial missing fraction:
  - 25%.
- Record nominal gap mix:
  - 20% short;
  - 30% long;
  - 50% very long.
- Record required metrics:
  - R2;
  - regression slope;
  - RMSE;
  - bias;
  - day/night metrics;
  - EBR.
- Create `docs/method_spec.md`.
- Create `docs/supplement_benchmarks.md` that records which supplementary table/figure supports each numerical reproduction target.
- Add a section called `Known ambiguities`.

### Exit criterion

There is one concise specification document that the implementation can be checked against.

---

## Step 2 — Scaffold the package

### Goal

Create an installable modern Python package before implementing scientific logic.

### Tasks

Create:

```text
pyproject.toml
src/rfrgapfill/
tests/
examples/
README.md
LICENSE
CITATION.cff
```

Recommended runtime dependencies:

```text
numpy
pandas
scipy
scikit-learn
joblib
```

Recommended development dependencies:

```text
pytest
pytest-cov
ruff
mypy
```

Do not require notebooks for the package core.

### Exit criterion

The empty package installs successfully and:

```bash
pytest
```

runs successfully.

---

## Step 3 — Implement schemas and validated configuration

### Goal

Avoid hard-coded FLUXNET names and scattered constants.

### Tasks

Implement:

```python
RFRConfig
FeatureConfig
GapScenarioConfig
ValidationConfig
ColumnMap
```

Support:

```text
mode = RFR3 | RFR10
frequency
latitude or hemisphere
random_state
n_jobs
radiation thresholds
daytime radiation threshold
min observed fraction
gap allocation basis
feature mode
hyperparameter grid
```

Define canonical variable keys:

```text
shortwave
vpd
air_temperature
net_radiation
wind_speed
wind_direction
soil_heat_flux
soil_temperature
relative_humidity
soil_water_content
```

Map these to arbitrary input columns.

### Tests

- reject unknown modes;
- reject missing RFR3/RFR10 required mappings;
- reject invalid thresholds;
- reject invalid gap proportions;
- test latitude-to-hemisphere logic.

### Exit criterion

All package behavior needed later can be represented in configuration without editing source constants.

---

## Step 4 — Build timestamp and cadence utilities

### Goal

Make temporal behavior correct before gap logic is added.

### Tasks

Implement utilities to:

- validate a timestamp column or `DatetimeIndex`;
- sort data;
- detect duplicates;
- infer time step;
- verify regular cadence;
- calculate elapsed hours from the first timestamp;
- translate durations such as `24h`, `7d`, and `30d` to timestamp intervals.

Do not assume that one day always equals 48 rows.

### Tests

Use:

- 30-minute data;
- hourly data;
- missing timestamps;
- duplicate timestamps.

### Exit criterion

Temporal operations use actual elapsed time.

---

## Step 5 — Implement pure feature transformers

### Goal

Reproduce receptive-limiter feature engineering in small, testable functions.

### Tasks

Implement separately:

```python
radiation_tag(...)
time_distance_hours(...)
season_tag(...)
daily_flux_statistics(...)
build_feature_matrix(...)
```

For daily statistics return:

```text
Q1
Q2
Q3
STD
```

Implement hemisphere-specific seasons exactly.

Define and document boundary behavior for the radiation categories.

### Tests

Hand-calculate expected outputs for small fixtures.

Test radiation at:

```text
9.9
10.0
50.0
100.0
100.1
```

Test every month in both hemispheres.

### Exit criterion

Feature engineering works without requiring a Random Forest model.

---

## Step 5A — Implement the ORF baseline

### Goal

Reproduce the supplementary comparison between RFR3 and Random Forest without the receptive limiter.

### Tasks

Support an explicit option such as:

```python
use_receptive_limiter=True
```

and:

```python
use_receptive_limiter=False
```

Define the latter as **ORF**.

ORF must:

- use the same Random Forest estimator family;
- use the same base meteorological driver set as the corresponding RFR run;
- omit the receptive-limiter-derived features;
- use the same train/test gap mask when compared with RFR;
- use the same random seed and hyperparameter-search policy where possible.

Return paired ORF/RFR metrics:

```text
R2
slope
RMSE
bias
```

Do not assert that RFR must improve every metric for every individual site.

### Exit criterion

A reproducible paired ORF-versus-RFR validation can be run on the same artificial gaps.

---

## Step 6 — Make target-derived features leakage safe

### Goal

Prevent hidden validation truth from being used to construct its own predictors.

### Tasks

Design `daily_flux_statistics()` to accept an `available_mask` or equivalent.

For artificial-gap validation:

1. build the holdout mask;
2. mask held-out target values;
3. compute target-derived daily statistics only from remaining available observations;
4. build prediction features;
5. retain untouched truth separately only for scoring.

Define behavior for a day with too few or zero target observations.

Possible explicit strategies:

```text
within_day_available
neighbor_day_fallback
rolling_available
missing
```

The default strategy must be documented and must not use hidden truth.

If exact legacy behavior is added later, isolate it:

```python
feature_mode="legacy_fluxlib"
```

### Required leakage test

- create an artificial gap;
- save prediction features;
- alter the hidden target values by a very large amount;
- rebuild features;
- verify the features for held-out rows do not change.

### Exit criterion

A dedicated automated test proves validation leakage cannot occur in the default mode.

---

## Step 7 — Implement RFR model fitting

### Goal

Create a low-level model class independent of gap-generation logic.

### Tasks

Wrap:

```python
sklearn.ensemble.RandomForestRegressor
```

Implement:

```python
fit(X, y)
predict(X)
get_feature_names()
get_best_params()
save()
load()
```

Use:

```python
GridSearchCV
```

but keep the hyperparameter grid in configuration.

Do not label an arbitrary grid as the exact Zhu et al. grid because the paper does not provide it.

Support a named compatibility preset later if implementation evidence is available.

Set deterministic seeds.

### Tests

- fit on synthetic nonlinear data;
- deterministic results;
- save/load equality;
- useful error for insufficient training data.

### Exit criterion

A fitted model can be trained and restored without knowing anything about artificial gaps.

---

## Step 8 — Build the high-level fit/fill API

### Goal

Provide a safe operational interface for real missing data.

### Tasks

Implement something similar to:

```python
filler = RFRGapFiller(config)
filler.fit(df, target=..., column_map=..., qc_col=...)
result = filler.fill(df)
```

Rules:

- train on eligible observed target rows only;
- preserve original target data;
- predict only target gaps;
- do not silently fill rows lacking required predictors;
- return provenance.

Recommended result fields:

```text
<target>_original
<target>_filled
<target>_is_observed
<target>_is_filled
<target>_fill_method
```

### Tests

- no observed values changed;
- missing values filled when predictors exist;
- rows with missing required drivers remain unfilled or raise according to configuration;
- input DataFrame is not mutated unless explicitly requested.

### Exit criterion

The package can fill an ordinary real-world missing interval.

---

## Step 9 — Implement the artificial-gap generator

### Goal

Reproduce the paper's validation scenario.

### Tasks

Create a `GapScenarioGenerator`.

Inputs should include:

```python
missing_fraction=0.25
durations={
    "short": "24h",
    "long": "7d",
    "very_long": "30d",
}
proportions={
    "short": 0.20,
    "long": 0.30,
    "very_long": 0.50,
}
min_observed_fraction=0.50
allocation_basis="missing_records"
random_state=42
```

Requirements:

- proposed intervals are chosen randomly;
- duration is determined by timestamps;
- reject intervals with insufficient measured target coverage;
- avoid endless retries;
- return a gap manifest;
- optionally disallow overlap among artificial gaps;
- support one common gap mask for NEE/H/LE.

### Gap manifest fields

At minimum:

```text
gap_id
gap_class
start
end
duration
n_expected
n_observed_before_masking
observed_fraction
```

### Tests

Validate exact temporal duration and seed reproducibility.

### Exit criterion

A deterministic artificial-gap scenario can be generated and inspected independently of model fitting.

---

## Step 10 — Resolve and expose the gap-allocation ambiguity

### Goal

Do not bury an interpretation of the paper's 20/30/50 percentages.

### Tasks

Support:

```python
allocation_basis="missing_records"
```

and, if feasible:

```python
allocation_basis="gap_events"
```

For each mode:

- write the algorithm explicitly;
- report achieved fraction and mix;
- warn when dataset length prevents the requested design.

Prefer `missing_records` as the initial default because the paper frames the scenario around percentages of removed half-hours, but document this as an implementation choice rather than an unambiguous fact.

### Tests

Confirm the reported proportions are within a defined tolerance.

### Exit criterion

Validation reports state exactly how the gap mixture was constructed.

---

## Step 11 — Implement metrics

### Goal

Match the paper's evaluation quantities.

### Tasks

Implement:

```python
r2(...)
regression_slope(...)
rmse(...)
bias(...)
energy_balance_ratio(...)
```

Bias:

```python
(sum(predicted) - sum(measured)) / n
```

Slope should use:

```text
x = measured
y = predicted
```

EBR:

```python
sum(H + LE) / sum(NETRAD - G)
```

Handle empty subsets and zero EBR denominator safely.

### Tests

Use fixtures with values for which every metric can be calculated by hand.

### Exit criterion

Metric functions are independent, documented, and numerically tested.

---

## Step 12 — Add daytime/nighttime evaluation

### Goal

Reproduce the paper's day/night comparison.

### Tasks

Define:

```python
day = shortwave > 20.0
night = shortwave <= 20.0
```

Make the threshold configurable.

For each validation run return metrics for:

```text
all
day
night
```

### Tests

Include timestamps on both sides of the 20 W m-2 threshold and at the boundary.

### Exit criterion

Validation output contains all/day/night statistics.

---

## Step 13 — Build paper-validation orchestration

### Goal

Create one method that performs the entire artificial-gap experiment correctly.

### Suggested flow

```text
original data
    ↓
identify genuinely measured target rows
    ↓
generate artificial gap manifest/mask
    ↓
save hidden truth separately
    ↓
mask artificial gaps
    ↓
build leakage-safe features
    ↓
fit/tune model on remaining observations
    ↓
predict artificial gaps
    ↓
compare prediction with hidden truth
    ↓
calculate all/day/night metrics
    ↓
calculate EBR where applicable
    ↓
return predictions + metrics + manifest + configuration
```

Implement something like:

```python
report = validate_rfr(
    df,
    targets=["NEE", "H", "LE"],
    mode="RFR10",
    scenario="zhu2022",
)
```

Do not use a random row-wise 25% holdout as a substitute for contiguous temporal gaps.

### Exit criterion

One call can execute a transparent end-to-end Zhu-style validation.

---

## Step 14 — Add FLUXNET2015 adapter utilities

### Goal

Make reproducing the published workflow easier without coupling the core package to FLUXNET.

### Tasks

Provide optional helpers for the paper's names:

```text
NEE_VUT_REF
NEE_VUT_REF_QC
H_F_MDS
H_F_MDS_QC
LE_F_MDS
LE_F_MDS_QC
SW_IN_F
VPD_F_MDS
TA_F_MDS
NETRAD
WS
WD
G_F_MDS
TS_F_MDS
RH
SWC_F_MDS
```

Implement helpers to:

- construct `ColumnMap`;
- interpret QC flags only after documenting FLUXNET flag semantics;
- report missing required variables.

Do not embed download credentials or site-specific assumptions.

### Exit criterion

A FLUXNET DataFrame can be mapped to the generic package API in a few lines.

---

## Step 15 — Add model provenance and run manifests

### Goal

Make every result auditable.

### Tasks

Store:

```text
package version
Python version
scikit-learn version
target
RFR mode
feature mode
column map
time resolution
hemisphere/latitude
random seed
hyperparameter grid
best parameters
gap configuration
gap manifest
training-row count
prediction-row count
QC rules
```

Support JSON export.

### Exit criterion

A gap-filled result can be traced back to the exact settings that produced it.

---

## Step 16 — Add synthetic scientific fixtures

### Goal

Test behavior on data with known structure.

### Tasks

Create at least one year of synthetic 30-minute data containing:

- solar-radiation diurnal cycle;
- air-temperature seasonality;
- VPD relationship;
- soil-temperature lag;
- synthetic LE/H/NEE relationships;
- random measurement noise.

Then generate known:

- 24-hour gaps;
- 7-day gaps;
- 30-day gaps.

Keep the fixture generation deterministic.

### Exit criterion

Tests do not depend on downloading external data.

---

## Step 17 — Run end-to-end RFR3 and RFR10 validation

### Goal

Verify both published model configurations.

### Tasks

On synthetic data:

1. run RFR3;
2. run RFR10;
3. output:
   - R2;
   - slope;
   - RMSE;
   - bias;
   - day/night results;
   - metrics by gap class;
   - gap manifest.
4. verify observed values remain unchanged.

Do **not** require RFR10 to outperform RFR3 in every synthetic test; that is not a valid software assertion.

### Exit criterion

Both workflows run successfully and deterministically.

---

## Step 18 — Validate gap-length sensitivity reporting

### Goal

Support the key scientific comparison in the paper.

### Tasks

Aggregate validation results separately for:

```text
short
long
very_long
```

Return a tidy DataFrame like:

```text
target | mode | gap_class | subset | r2 | slope | rmse | bias | n
```

Provide plotting helpers only after the data tables are correct.

### Published benchmark targets

Document Supplementary Table S3 medians for reproduction runs.

At minimum include:

```text
Diel median R2
              MDS   RFR3  RFR10
NEE           0.72  0.78   0.84
H             0.67  0.78   0.90
LE            0.63  0.75   0.85
```

and very-long-gap medians:

```text
NEE R2:  MDS=0.59, RFR3=0.74, RFR10=0.74
H   R2:  MDS=0.62, RFR3=0.76, RFR10=0.89
LE  R2:  MDS=0.57, RFR3=0.74, RFR10=0.83
```

These are reproduction benchmarks, not universal acceptance thresholds for arbitrary stations.

### Exit criterion

Users can directly compare model performance by gap duration and compare matched FLUXNET reproduction runs against the published distributions.

---

## Step 18A — Add supplementary uncertainty diagnostics

### Goal

Expose the additional gap-length uncertainty behavior documented in Supplementary Table S8 and the companion figure.

### Tasks

Implement immediately:

```python
bias_iqr(...)
```

and report bias IQR by:

```text
target
method
gap_class
IGBP, when available
```

Document the published very-long-gap bias-IQR / joint-uncertainty-CI ranges:

```text
NEE: MDS 5.21–5.34, RFR3 2.90–2.97, RFR10 0.87–0.89
H:   MDS 5.37–5.63, RFR3 2.46–2.57, RFR10 1.65–1.72
LE:  MDS 7.90–8.93, RFR3 2.25–2.55, RFR10 1.79–2.02
```

Do **not** implement a supposedly exact normalized joint-uncertainty metric until the denominator definition has been independently verified.

If implemented later, call it experimental until reproduced against Table S8.

### Exit criterion

Bias-IQR by gap class is available, and the normalized uncertainty metric cannot be mistaken for an exact reproduction unless its formula has been verified.

---

## Step 19 — Implement EBR validation for H and LE

### Goal

Reproduce the paper's independent energy-balance check.

### Tasks

For artificial gaps where required inputs are available, calculate:

```text
EBR_measured
EBR_filled
EBR_difference
```

Use the same intervals for measured and filled calculations.

Validate the denominator and missing-data handling.

### Exit criterion

Energy-balance validation is included in the report for H/LE workflows.

---

## Step 20 — Optional legacy `fluxlib` compatibility audit

### Goal

Improve historical reproducibility without compromising the paper-faithful default.

### Tasks

Only after the core package works:

- inspect the paper-era `fluxlib` source referenced by Zhu et al.;
- document any implementation behavior not stated in the article;
- compare it with the new implementation;
- classify differences as:
  - paper-consistent;
  - paper-ambiguous;
  - paper-conflicting;
  - later repository change.

If worthwhile, implement:

```python
feature_mode="legacy_fluxlib"
hyperparameter_preset="legacy_fluxlib"
```

Keep these modes explicitly named.

Do not replace `paper_safe` defaults with legacy behavior merely to match old source code.

### Exit criterion

Historical compatibility, if provided, is isolated and documented.

---

## Step 20A — Add multi-site and ecosystem-stratified reproduction reports

### Goal

Use Supplementary Tables S1-S6 and S9 as structured validation context.

### Tasks

For multi-site validation, include where available:

```text
site_id
latitude
longitude
IGBP
Koppen
instrument system
included_in_94_site_subset
```

Support tidy reports grouped by:

```text
site
target
method
gap_class
day/night
IGBP
```

Important scientific constraints:

- RFR3 has broader validation evidence than RFR10.
- Supplementary Table S9 compares MDS and RFR3 for NEE across 194 sites.
- The full NEE/H/LE RFR3/RFR10 comparison is based on a 94-site subset.
- Do not require every site to show improvement.
- Avoid universal performance thresholds across ecosystems.

Optionally reproduce the Welch-test comparisons from Supplementary Table S10 across sites.

### Exit criterion

The package can create site- and ecosystem-stratified reports without pooling sites into one predictive model.

---

## Step 21 — Documentation

### Goal

Make the package usable by someone who did not write it.

### README must include

- scientific purpose;
- citation to Zhu et al. (2022);
- installation;
- expected input format;
- RFR3/RFR10 drivers;
- quick-start example;
- real-gap filling example;
- artificial-gap validation example;
- explanation of output provenance;
- explanation of leakage-safe target statistics;
- known differences/ambiguities relative to the paper.

### Additional docs

Create:

```text
docs/method.md
docs/validation.md
docs/assumptions.md
docs/fluxnet.md
docs/supplement_benchmarks.md
```

### Exit criterion

A new user can run the synthetic example from the README.

---

## Step 22 — Add a CLI only after the Python API is stable

### Goal

Support reproducible batch workflows without making CLI design drive the library design.

### Example

```bash
rfrgapfill validate data.csv \
  --target LE \
  --mode RFR10 \
  --config config.yaml \
  --output validation/
```

and:

```bash
rfrgapfill fill data.csv \
  --target LE \
  --mode RFR10 \
  --config config.yaml \
  --output filled.csv
```

### Exit criterion

CLI invokes the same tested Python API rather than duplicating logic.

---

## Step 23 — Add CI and quality checks

### Goal

Prevent scientific logic from drifting silently.

### CI should run

```text
pytest
coverage
ruff
package build
README smoke example
```

Optionally add static typing.

Test on multiple supported Python versions.

### Exit criterion

Every pull request receives automatic validation.

---

## Step 24 — Release checklist

Before version `0.1.0`:

- [ ] RFR3 implemented.
- [ ] RFR10 implemented.
- [ ] ORF/no-receptive-limiter baseline implemented.
- [ ] radiation tag tested.
- [ ] time-distance feature tested.
- [ ] Northern and Southern season tags tested.
- [ ] daily Q1/Q2/Q3/std tested.
- [ ] leakage test passes.
- [ ] 24-hour artificial gaps tested.
- [ ] 7-day artificial gaps tested.
- [ ] 30-day artificial gaps tested.
- [ ] shared multi-target gap mask tested.
- [ ] R2 tested.
- [ ] slope tested.
- [ ] RMSE tested.
- [ ] bias tested.
- [ ] day/night metrics tested.
- [ ] EBR tested.
- [ ] Bias-IQR by gap class tested.
- [ ] Supplementary Table S3 benchmark values documented.
- [ ] Site/IGBP metadata can be retained in multi-site reports.
- [ ] random-seed reproducibility tested.
- [ ] save/load tested.
- [ ] source distribution and wheel build.
- [ ] clean-environment install tested.
- [ ] README example executes.
- [ ] citation file included.
- [ ] scientific ambiguities documented.

---

# Working rules for the AI agent

Throughout implementation:

1. **Read the method specification before each major module.**
2. **Implement one concept at a time.**
3. **Write or update tests in the same step as the code.**
4. **Run the relevant tests after every change.**
5. **Never use hidden validation target values as input features.**
6. **Never overwrite observed fluxes during filling unless explicitly requested.**
7. **Never silently impute missing meteorological drivers.**
8. **Never call an undocumented implementation choice "Zhu et al. exact."**
9. **Use configuration for ambiguous choices.**
10. **Keep scientific calculations separate from plotting and I/O.**
11. **Record random seeds and model configuration with every validation result.**
12. **Prefer clear, inspectable intermediate tables over opaque pipelines.**
13. **Treat the paper-faithful implementation and future algorithmic enhancements as separate modes.**
14. **Do not optimize performance before correctness and reproducibility are established.**
15. **Do not declare the package complete until the leakage and gap-duration tests pass.**
16. **Treat the supplementary tables as reproduction benchmarks, not universal accuracy thresholds.**
17. **Keep ORF, RFR3, and RFR10 comparable by using matched gap masks and seeds.**
18. **Do not call normalized uncertainty metrics exact until the Table S8 denominator has been verified.**
