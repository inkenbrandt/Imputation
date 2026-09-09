# Context for an AI Coding Agent: Reproduce Zhu et al. (2022) RFR Eddy-Covariance Gap Filling

## 1. Mission

Build a production-quality Python package that reproduces, as faithfully and transparently as possible, the **Random Forest Robust (RFR)** eddy-covariance gap-filling strategy described by:

> Zhu, S., Clement, R., McCalmont, J., Davies, C. A., & Hill, T. (2022). *Stable gap-filling for longer eddy covariance data gaps: A globally validated machine-learning approach for carbon dioxide, water, and energy fluxes*. Agricultural and Forest Meteorology, 314, 108777. https://doi.org/10.1016/j.agrformet.2021.108777

The intended package should be useful for practical eddy-covariance processing, while retaining a **paper-faithful mode** that implements the method described in the publication.

The paper states that its RFR implementation was developed in the `fluxlib` Python package and was based on scikit-learn Random Forests. The paper points to:

- https://github.com/soonyenju/fluxlib

The package created in this project should be an independent, well-tested implementation rather than a direct copy.

The accompanying journal supplements are part of the scientific specification for this project. They provide site metadata, aggregate benchmarks, site-level validation tables, uncertainty summaries, statistical comparisons, and a direct comparison of RFR3 against ordinary Random Forest without the receptive limiter.

---

## 2. Scientific objective

The package must fill missing eddy-covariance flux observations using Random Forest regression and the feature-engineering strategy described by Zhu et al.

Primary target fluxes in the paper are:

- **NEE** — net ecosystem exchange of CO2
- **H** — sensible heat flux
- **LE** — latent heat flux

The API should remain generic enough to gap-fill other continuous flux variables if appropriate drivers are supplied.

The method was designed especially to remain stable for **longer gaps**, including gaps on the order of days to one month.

---

## 3. RFR variants

Implement two named configurations.

### RFR3

Uses the same three meteorological drivers used by the paper's MDS benchmark:

1. Downward shortwave radiation — FLUXNET name: `SW_IN_F`
2. Vapor pressure deficit — `VPD_F_MDS`
3. Air temperature — `TA_F_MDS`

### RFR10

Uses all RFR3 drivers plus:

4. Net radiation — `NETRAD`
5. Wind speed — `WS`
6. Wind direction — `WD`
7. Soil heat flux — `G_F_MDS`
8. Soil temperature — `TS_F_MDS`
9. Relative humidity — `RH`
10. Soil water content — `SWC_F_MDS`

Do not hard-code the package to these FLUXNET column names. Provide a column-mapping/schema layer so users can map their own station names to the canonical variables.

---

## 4. Input-data assumptions

The original analysis used half-hourly FLUXNET2015 data. The package should:

- support a `DatetimeIndex` or explicit timestamp column;
- infer or accept a time step explicitly;
- work correctly at 30-minute resolution;
- preferably support other regular resolutions, such as hourly data;
- calculate gap lengths using elapsed time, not only row counts;
- require timestamps to be monotonic and unique unless a documented preprocessing option resolves duplicates;
- distinguish **observed flux measurements** from values that were already gap-filled before ingestion when a QC/provenance field is available.

For FLUXNET compatibility, the paper used quality-control flags to identify data that had already been gap-filled. Training and validation should preferentially use genuinely observed, quality-controlled flux data.

Environmental driver variables may already be gap-filled. The paper used pre-filled FLUXNET meteorological drivers.

---

## 5. Train separate models by site and target

The paper trained RFR separately for each site.

The package should therefore default to:

- one site at a time;
- one target flux at a time;
- a separate fitted model for NEE, H, LE, or another requested target.

Do **not** pool data from different sites in the default paper-faithful workflow.

Multi-site orchestration may be added as a convenience layer that loops over independent site-level models.

---

## 6. Feature engineering: the "receptive limiter"

The paper identifies its feature-engineering stage, called the **receptive limiter**, as a central part of RFR.

Implement feature engineering as deterministic, independently testable transformers.

### 6.1 Original environmental drivers

Include the selected RFR3 or RFR10 driver variables.

### 6.2 Radiation category

Categorize downward shortwave radiation into classes. The paper gives the following example thresholds:

- `weak`: radiation `< 10 W m-2`
- `medium`: `10–100 W m-2`
- `strong`: `> 100 W m-2`

The exact handling of values equal to 10 or 100 W m-2 is not perfectly explicit in the prose. Choose and document a deterministic boundary convention. Prefer inclusive, exhaustive bins such as:

- weak: `< 10`
- medium: `>= 10 and <= 100`
- strong: `> 100`

Expose thresholds in configuration so exact legacy behavior can be reproduced later if needed.

### 6.3 Time distance

Create a continuous feature representing **hours since the beginning of the time series**.

Example:

```text
timestamp                  time_distance_hours
2025-01-01 00:00                0.0
2025-01-01 00:30                0.5
2025-01-01 01:00                1.0
```

Use actual timestamp differences rather than row position.

The purpose is to represent gradual ecosystem growth, degradation, or other long-term temporal trends.

### 6.4 Seasonal category

Create season tags from calendar month and hemisphere.

Northern Hemisphere:

- winter: December–February
- spring: March–May
- summer: June–August
- autumn: September–November

Southern Hemisphere:

- winter: June–August
- spring: September–November
- summer: December–February
- autumn: March–May

Require the hemisphere, latitude, or a documented inference rule.

### 6.5 Daily target-flux statistics

For each target flux, derive daily distributional statistics from quality-controlled flux data:

- first quartile, Q1 / 25th percentile;
- median, Q2 / 50th percentile;
- third quartile, Q3 / 75th percentile;
- standard deviation.

The paper describes these as target-specific RFR input features intended to reduce the effect of potential outliers.

These statistics should be joined back to all timestamps belonging to the corresponding day.

### 6.6 Critical leakage requirement

Target-derived daily statistics create a major reproducibility and validation risk.

For **paper validation using artificial gaps**, the feature values for a held-out interval must not be allowed to use the hidden measured target values from that same interval unless the historical implementation is intentionally being reproduced in a clearly labeled compatibility mode.

The default implementation must therefore be **leakage safe**:

- create the artificial-gap mask before calculating target-derived features for validation;
- calculate daily target statistics using only target observations considered available to the model;
- never use held-out truth in features used to predict that truth;
- write an explicit unit test proving this.

If an exact historical implementation is later found to derive statistics differently, implement that separately as something like:

```text
feature_mode="legacy_fluxlib"
```

while keeping:

```text
feature_mode="paper_safe"
```

as the documented default.

Do not silently mix the two approaches.


### 6.7 Ordinary Random Forest (ORF) benchmark

The supplementary material includes **Supplementary Figure S1**, which compares RFR3 with a Random Forest that does **not** use the receptive-limiter feature engineering. The supplement calls this baseline **ORF**.

The package should therefore make the feature-engineering contribution directly testable.

Recommended configuration:

```python
use_receptive_limiter=True   # RFR behavior
use_receptive_limiter=False  # ORF benchmark
```

ORF should use the same underlying Random Forest machinery and the same meteorological driver set as the comparison RFR model, but omit the receptive-limiter-derived features. This makes it possible to determine whether improvements arise from the Random Forest itself or from the feature-engineering strategy.

Do not redefine ORF to mean a different estimator, parameter grid, or training dataset.

---

## 7. Model

Use `sklearn.ensemble.RandomForestRegressor`.

Requirements:

- deterministic `random_state`;
- configurable `n_jobs`;
- hyperparameter optimization with `GridSearchCV`;
- model trained only on rows with a valid target and valid required features;
- clear reporting of rows excluded due to missing predictors;
- separate model and feature transformer for each target;
- model/feature configuration serializable with `joblib`.

The paper states that hyperparameters were automatically optimized with `GridSearchCV`, but the article does not enumerate the complete search grid. Therefore:

1. provide a sensible documented default search grid;
2. isolate the grid in configuration;
3. if reproducing the archived paper-era `fluxlib` implementation, add its parameter grid as a named preset rather than pretending it came directly from the paper text.

Never invent an undocumented "paper exact" grid.

---

## 8. Artificial-gap validation scenario

Reproduce the paper's artificial-gap experiment as a first-class validation workflow.

### 8.1 Total withheld data

Artificial gaps should collectively remove approximately **25% of the total half-hourly observations** available for validation.

### 8.2 Gap lengths

Use three gap-duration classes:

- short: **24 hours**
- long: **7 days**
- very long: **30 days**

### 8.3 Allocation

The paper states that the artificial-gap scenario was composed of:

- 20% short gaps
- 30% long gaps
- 50% very-long gaps

The wording leaves some ambiguity as to whether these percentages refer to the number of gap events or the number of withheld half-hours.

Do not hide this ambiguity.

Implement a configurable option such as:

```python
allocation_basis="missing_records"  # preferred default
# or
allocation_basis="gap_events"
```

and document which interpretation is used in each result.

### 8.4 Shared gaps across targets

When jointly validating NEE, H, and LE, use the **same artificial gap locations** for all three targets, matching the paper.

### 8.5 Existing real gaps

When an artificial gap intersects existing missing data, require at least **50% genuinely observed target measurements** within the proposed interval. Otherwise reject the interval and sample a new location.

Make this threshold configurable:

```python
min_observed_fraction=0.50
```

### 8.6 Gap generation safeguards

The generator should:

- work from timestamps and requested durations;
- avoid impossible intervals at dataset boundaries;
- avoid infinite retry loops;
- report when the requested scenario cannot be constructed;
- preferably prevent artificial intervals from overlapping each other unless explicitly allowed;
- support seeded reproducibility;
- return a complete gap manifest with start, end, duration, class, affected rows, and observed fraction.

---

## 9. Training/test design for paper validation

Figure 2 of the paper depicts a 75% training set and 25% test set associated with the gap scenario.

For the paper-validation workflow:

- the **artificially withheld observations are the test set**;
- the remaining eligible observations are training data;
- do not perform a naive random row-wise split that destroys the intended temporal gap structure;
- tune Random Forest hyperparameters using only the training portion;
- avoid temporal/test leakage during cross-validation where practical.

A robust modern implementation may offer blocked or time-aware cross-validation in addition to conventional `GridSearchCV` folds. Any deviation from the published setup must be labeled as an enhancement rather than paper reproduction.

---

## 10. Gap filling of real missing data

For operational filling:

1. fit the model on eligible observed target rows;
2. identify rows where the target is missing;
3. calculate all prediction features without using unavailable target truth;
4. predict only rows with sufficient features;
5. preserve all original observed values;
6. place predictions only into missing locations;
7. provide a provenance column.

Recommended output fields:

```text
LE_original
LE_filled
LE_is_observed
LE_is_filled
LE_fill_method
LE_model_version
```

Optional outputs:

- per-tree prediction distribution;
- prediction standard deviation across trees;
- quality flag indicating extrapolation or incomplete features.

Do not overwrite the user's original target values in-place by default.

---

## 11. Validation metrics

Implement metrics comparing predictions against known measurements inside artificial gaps.

### Required

- coefficient of determination, `R2`;
- linear-regression slope of filled versus measured;
- root mean squared error, `RMSE`;
- bias.

Paper bias:

```text
bias = [sum(filled) - sum(measured)] / n
```

which is equivalent to mean prediction error for equal-weighted observations.

Be explicit about regression orientation when computing slope:

```text
measured -> x
filled   -> y
```

### Day/night subsets

The paper defines daytime as:

```text
downward shortwave radiation > 20 W m-2
```

Evaluate metrics for:

- all observations;
- daytime;
- nighttime.

### Energy-balance ratio

For H and LE, implement:

```text
EBR = sum(H + LE) / sum(NETRAD - G)
```

Calculate EBR for the same artificial-gap intervals using:

- measured H and LE;
- RFR-filled H and LE.

Return both values and their difference.

---


## 11A. Supplement-derived numerical validation benchmarks

The supplementary tables provide concrete numerical targets that should be used as **integration-test and reproduction benchmarks**, not as universal pass/fail requirements for arbitrary sites.

### 11A.1 Diel median performance across the 94-site complete-analysis subset

Supplementary Table S3 reports the following medians across sites.

| Flux | Method | Median R2 | Median slope | Median RMSE | Median bias |
|---|---|---:|---:|---:|---:|
| NEE | MDS | 0.72 | 0.79 | 2.98 | 0.00 |
| NEE | RFR3 | 0.78 | 0.79 | 2.63 | -0.01 |
| NEE | RFR10 | 0.84 | 0.84 | 2.42 | 0.02 |
| H | MDS | 0.67 | 0.72 | 50.33 | -1.12 |
| H | RFR3 | 0.78 | 0.78 | 39.37 | 0.26 |
| H | RFR10 | 0.90 | 0.90 | 26.80 | -0.13 |
| LE | MDS | 0.63 | 0.70 | 39.84 | -2.36 |
| LE | RFR3 | 0.75 | 0.76 | 33.50 | -0.11 |
| LE | RFR10 | 0.85 | 0.85 | 25.51 | -0.46 |

RMSE and bias are reported in:

- `g C m-2 d-1` for NEE in Table S3;
- `W m-2` for H and LE.

These values are useful for validating aggregate reproduction with matching FLUXNET inputs and matching preprocessing. Do not require a new dataset to match them.

### 11A.2 Very-long-gap median benchmarks

Supplementary Table S3 also provides useful medians for the longest artificial gaps.

#### NEE, very-long gaps

| Method | R2 | slope | RMSE | bias |
|---|---:|---:|---:|---:|
| MDS | 0.59 | 0.85 | 2.99 | -0.48 |
| RFR3 | 0.74 | 0.97 | 2.61 | -0.14 |
| RFR10 | 0.74 | 0.97 | 2.48 | -0.04 |

#### H, very-long gaps

| Method | R2 | slope | RMSE | bias |
|---|---:|---:|---:|---:|
| MDS | 0.62 | 0.88 | 55.00 | -9.41 |
| RFR3 | 0.76 | 0.99 | 40.11 | 0.66 |
| RFR10 | 0.89 | 1.00 | 26.80 | -0.20 |

#### LE, very-long gaps

| Method | R2 | slope | RMSE | bias |
|---|---:|---:|---:|---:|
| MDS | 0.57 | 0.85 | 45.45 | -20.24 |
| RFR3 | 0.74 | 0.98 | 34.59 | -0.35 |
| RFR10 | 0.83 | 0.99 | 25.49 | -3.66 |

These benchmarks reinforce that the extended RFR10 driver set is especially valuable for H and LE at long gap durations.

### 11A.3 Nighttime benchmark behavior

The supplements show that nighttime predictions remain much harder than daytime predictions.

Useful median examples from Table S3:

```text
H nighttime:
MDS    R2=0.03, slope=0.16, RMSE=29.88
RFR3   R2=0.16, slope=0.20, RMSE=19.46
RFR10  R2=0.49, slope=0.47, RMSE=14.24

LE nighttime:
MDS    R2=0.02, slope=0.17, RMSE=21.93
RFR3   R2=0.10, slope=0.15, RMSE=12.75
RFR10  R2=0.27, slope=0.32, RMSE=9.79
```

The package must not hide weak nighttime skill by reporting only aggregate metrics.

### 11A.4 Statistical comparison benchmarks

Supplementary Table S10 reports mean method differences and Welch-test 95% confidence intervals. Selected mean R2 improvements are:

```text
RFR3 - MDS:
NEE   +0.07
H     +0.11
LE    +0.11

RFR10 - RFR3:
NEE   +0.05
H     +0.12
LE    +0.10
```

The strongest and most consistent gains occur for H and LE. Some NEE differences, especially bias and RMSE differences, are not statistically significant according to the supplementary table.

The package may optionally reproduce these Welch tests across sites, but statistical testing should be separate from ordinary single-site gap filling.

---

## 11B. Supplement-derived uncertainty diagnostics

Supplementary Table S8 and the companion sensitivity figure compare gap-filling bias uncertainty against:

1. a flux confidence interval; and
2. a joint flux-uncertainty confidence interval.

The normalized uncertainty increases strongly with gap length for MDS and more gradually for RFR.

For very-long gaps, the reported bias-IQR / joint-uncertainty-CI intervals are approximately:

| Flux | MDS | RFR3 | RFR10 |
|---|---|---|---|
| NEE | 5.21–5.34 | 2.90–2.97 | 0.87–0.89 |
| H | 5.37–5.63 | 2.46–2.57 | 1.65–1.72 |
| LE | 7.90–8.93 | 2.25–2.55 | 1.79–2.02 |

The package should support basic bias-IQR reporting by gap class immediately.

However, **do not claim exact reproduction of the normalized joint-uncertainty ratios until the denominator calculation is reconstructed and verified from the supplementary methods or source implementation**. Treat this as an optional research/reproduction module rather than a core gap-filling dependency.

---

## 11C. Site and ecosystem stratification from the supplements

Supplementary Tables S1 and S2 describe the broader FLUXNET site population and identify the 94-site complete-analysis subset.

For reproduction studies, preserve metadata such as:

```text
site_id
latitude
longitude
continent
country
IGBP
Koppen climate class
start_date
end_date
elevation
instrument system
instrument-to-canopy height ratio
included_in_94_site_subset
```

The wider metadata set contains 194 sites spanning multiple continents and 11 IGBP classes. The complete RFR3/RFR10/H/LE/NEE comparison used a 94-site subset because complete QC and RFR10 driver availability were more restrictive.

Supplementary Table S9 provides a broader 194-site NEE comparison for MDS versus RFR3. Therefore:

- do not treat the 94-site subset as the only validation evidence for RFR3;
- do not assume RFR10 can be run at every RFR3 site;
- report site and ecosystem metadata alongside multi-site validation results.

The supplementary results also show that model performance varies by ecosystem. Avoid a universal assertion such as `R2 > 0.8` for every site.

---

## 11D. Supplementary source map for the implementation project

Use the supplied supplements as follows:

```text
mmc1.docx  -> Supplementary Figure S1: RFR3 vs ORF / receptive-limiter effect
mmc2.docx  -> Gap-length sensitivity and normalized uncertainty discussion/figure
mmc3.docx  -> Supplementary Tables S1-S2: site metadata and 94-site membership
mmc4.docx  -> Supplementary Table S3: summary performance distributions
mmc5.docx  -> Supplementary Tables S4-S6: site-level diel/day/night results
mmc6.docx  -> Supplementary Table S8: normalized uncertainty ranges
mmc8.docx  -> Supplementary Table S9: 194-site NEE MDS vs RFR3 comparison
mmc9.docx  -> Supplementary Table S10: Welch-test confidence intervals
```

The uploaded `mmc7.xlsx` should be retained as a benchmark data asset, but its exact table identity and structure must be inspected before using it programmatically. Do not infer or hard-code its role from filename ordering alone.

## 12. Reproducibility controls

All stochastic components must accept a seed:

- artificial gap selection;
- Random Forest;
- cross-validation where applicable.

A validation run should record:

- package version;
- Python version;
- scikit-learn version;
- input column mapping;
- target;
- RFR3/RFR10 mode;
- feature-mode selection;
- hyperparameter search grid;
- chosen best parameters;
- random seed;
- gap manifest;
- time resolution;
- hemisphere/latitude;
- QC rules.

Prefer a serializable run manifest in JSON or YAML.

---

## 13. Proposed package architecture

Use a modern `src/` layout:

```text
rfr-gapfill/
├── pyproject.toml
├── README.md
├── LICENSE
├── CITATION.cff
├── src/
│   └── rfrgapfill/
│       ├── __init__.py
│       ├── config.py
│       ├── schema.py
│       ├── time.py
│       ├── features.py
│       ├── gaps.py
│       ├── model.py
│       ├── fill.py
│       ├── metrics.py
│       ├── validation.py
│       ├── fluxnet.py
│       ├── provenance.py
│       └── cli.py
├── tests/
│   ├── test_time.py
│   ├── test_features.py
│   ├── test_gaps.py
│   ├── test_model.py
│   ├── test_fill.py
│   ├── test_metrics.py
│   ├── test_validation.py
│   └── fixtures/
└── examples/
    ├── synthetic_example.py
    └── fluxnet_example.ipynb
```

Keep modules small and composable.

---

## 14. Recommended public API

A simple high-level interface should look approximately like:

```python
from rfrgapfill import RFRConfig, RFRGapFiller

config = RFRConfig(
    mode="RFR10",
    frequency="30min",
    hemisphere="north",
    random_state=42,
)

filler = RFRGapFiller(config)

filler.fit(
    df,
    target="LE",
    qc_col="LE_QC",
    column_map={
        "shortwave": "SW_IN",
        "vpd": "VPD",
        "air_temperature": "TA",
        "net_radiation": "Rn",
        "wind_speed": "WS",
        "wind_direction": "WD",
        "soil_heat_flux": "G",
        "soil_temperature": "TS",
        "relative_humidity": "RH",
        "soil_water_content": "SWC",
    },
)

result = filler.fill(df)
```

Validation:

```python
report = filler.validate(
    df,
    artificial_gaps=True,
    missing_fraction=0.25,
    gap_mix={
        "24h": 0.20,
        "7d": 0.30,
        "30d": 0.50,
    },
)
```

The exact API can change, but it should remain explicit, typed, and difficult to misuse.

---

## 15. Configuration objects

Use dataclasses or Pydantic-style validated configuration objects.

Suggested objects:

```text
RFRConfig
FeatureConfig
GapScenarioConfig
ColumnMap
ValidationConfig
```

Important configuration fields:

```text
mode
frequency
hemisphere or latitude
random_state
radiation_thresholds
daytime_threshold
feature_mode
hyperparameter_grid
min_observed_fraction
allocation_basis
cv_strategy
n_jobs
```

Validate incompatible options early.

---

## 16. Handling missing meteorological drivers

The paper used pre-filled drivers.

Default behavior should therefore be conservative:

- do not silently fabricate missing meteorological values;
- identify which required drivers are missing;
- either fail clearly or leave predictions missing for rows lacking predictors.

Optional driver interpolation/imputation can be added as a separate preprocessing component, but it must be explicit and should produce its own provenance flags.

---

## 17. Scientific ambiguities must remain visible

The article does not specify every implementation detail needed for bit-for-bit reconstruction.

Examples include:

- exact Random Forest hyperparameter grid;
- exact coding of radiation-category boundaries;
- precise interpretation of the 20/30/50 gap allocation percentages;
- exact handling of target-derived daily statistics when an entire day or longer interval is missing;
- cross-validation details inside `GridSearchCV`.

When the paper is silent:

1. do not invent behavior and call it "paper exact";
2. document the ambiguity;
3. select a defensible default;
4. expose the choice in configuration;
5. add compatibility modes only when implementation evidence exists.

---

## 18. Distinguish reproduction from enhancements

The package may include scientifically useful enhancements, but they must not alter the paper-faithful default invisibly.

Examples of enhancements:

- blocked/time-series cross-validation;
- uncertainty estimates from tree ensembles;
- additional satellite or vegetation drivers;
- alternate ML algorithms;
- multi-site models;
- automatic meteorological imputation;
- gap lengths longer than 30 days.

Place such behavior behind explicit options or separate classes.

---

## 19. Required tests and acceptance criteria

The implementation is not complete until the following pass.

### Feature tests

1. Radiation categories are correct below, at, between, and above thresholds.
2. Northern Hemisphere seasons are correct for all 12 months.
3. Southern Hemisphere seasons are correct for all 12 months.
4. Time-distance values equal actual elapsed hours.
5. Daily Q1/Q2/Q3/std values match hand-calculated values.
6. Feature names and order are deterministic.

### Receptive-limiter benchmark tests

7. ORF and RFR use the same Random Forest estimator family and same base driver set.
8. ORF omits receptive-limiter-derived features.
9. RFR includes the configured receptive-limiter features.
10. On a fixed reproducible benchmark dataset, record the difference between RFR and ORF for R2, slope, RMSE, and bias without asserting that every individual metric must improve.

### Leakage test

11. Hide a known artificial interval.
12. Alter the hidden truth values dramatically.
13. Recompute all features used to predict the hidden interval.
14. In `paper_safe` mode, prediction features and predictions must not change because hidden truth changed.

### Gap-generator tests

15. A 24-hour gap spans 24 elapsed hours at the configured cadence.
16. A 7-day gap spans 7 days.
17. A 30-day gap spans 30 days.
18. Generated gaps are reproducible with the same seed.
19. Proposed gaps failing the observed-fraction criterion are rejected.
20. Shared gap masks are identical across NEE, H, and LE in joint validation.
21. The overall withheld fraction and class allocation are within a documented tolerance.

### Model/filling tests

22. RFR3 requires exactly the three canonical driver types.
23. RFR10 requires the ten canonical driver types.
24. Observed target values are never replaced by default.
25. Only eligible missing rows receive model predictions.
26. Same data + config + seed produces identical predictions within numerical tolerance.
27. Saved and reloaded models reproduce predictions.

### Metric tests

28. R2 matches a hand-calculated fixture.
29. Regression slope matches a hand-calculated fixture.
30. RMSE matches a hand-calculated fixture.
31. Bias matches `(sum(pred)-sum(obs))/n`.
32. Day/night split uses the configured radiation threshold.
33. EBR matches `sum(H+LE)/sum(NETRAD-G)` on a hand-calculated fixture.

### End-to-end tests

34. Synthetic half-hourly data with known diurnal and seasonal structure can be trained, artificially gapped, filled, and scored.
35. End-to-end tests include 1-day, 7-day, and 30-day gaps.
36. Package builds a wheel and installs into a clean environment.
37. README quick-start code runs as written.
38. A FLUXNET reproduction test can compare aggregate RFR3/RFR10 metrics against Supplementary Table S3 without requiring exact equality unless preprocessing and source versions are identical.
39. Multi-site reports include site ID and, where available, IGBP metadata.
40. Gap-class reports can calculate bias IQR; normalized uncertainty ratios remain explicitly marked experimental until their denominator definition is verified.

---

## 20. Definition of done

The first release is complete when:

- RFR3 and RFR10 are both implemented;
- receptive-limiter features are implemented and tested;
- an ORF/no-receptive-limiter benchmark mode is implemented;
- artificial 1-, 7-, and 30-day validation gaps are reproducible;
- leakage-safe training/validation is enforced by default;
- metrics and EBR are available;
- published supplementary benchmark tables are documented for reproduction testing;
- gap-length bias-IQR reporting is available;
- observed-versus-filled provenance is preserved;
- the package is installable from `pyproject.toml`;
- the public API is documented;
- automated tests pass;
- a synthetic demonstration is included;
- all known divergences or ambiguities relative to Zhu et al. are documented.

The priority is **scientific reproducibility, explicit assumptions, and prevention of data leakage**, not merely achieving high predictive scores.
