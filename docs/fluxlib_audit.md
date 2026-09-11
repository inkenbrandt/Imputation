# `fluxlib` compatibility audit (Step 20)

Audit of the paper-era source of [`fluxlib`](https://github.com/soonyenju/fluxlib),
the implementation Zhu et al. (2022) cite for RFR, against the article (as frozen
in [`method_spec.md`](method_spec.md)) and against this package.

**Outcome in one paragraph.** The historical code builds the daily target
statistics from the whole series *before* the artificial gaps are applied, so
held-out values reach the features used to predict them (ambiguity A6 is
answered, in the leaky direction). It also computes seven statistics rather than
the paper's four, joins each day's statistics to the *previous* day's rows,
linearly interpolates the target across every gap first, and uses day-of-year and
year instead of the paper's elapsed-hours feature. The `GridSearchCV` grid exists
but no archived pipeline calls it. All of this is now available as an explicitly
named, warning-emitting compatibility mode, `feature_mode="legacy_fluxlib"`, plus
two named hyperparameter presets. **No `paper_safe` default was changed.**

---

## 1. What was inspected

Repository: <https://github.com/soonyenju/fluxlib>, cloned 2026-09-10, 146 commits.
The repository also carries archived copies of earlier releases under
`fluxlib-history-versions/`.

| Release / commit | Date | Role in this audit |
|---|---|---|
| 0.0.13 (archived snapshot) | 2020 | `Filler` API used by the archived artificial-gap notebooks |
| 0.0.14 - 0.0.17 (`77c9ce3` ... `74e5ec9`) | 2021-02 | `GFiller`, config-driven regressors, `make_gap_pipeline` |
| 0.0.21 - 0.0.23 (`c58ddbf`, `bc44a5d`, `da53256`) | 2021-05-31 to 2021-08-04 | **the paper-era reference**: 0.0.23 is the last change to the gap-filling logic before the article appeared |
| `ebdea23` | 2021-10-28 | adds `load_mds_txt` to `utils.py`; no behavioural change |
| 0.0.26 - 0.0.29 (`084915d` ... `dd0752e`) | 2022-09 to 2024-04 | post-publication |
| `d0fdccd` and neighbours | 2025-02 | the RFR gap-filling module is deleted and replaced by an XGBoost ensemble |

Files read at `da53256` (0.0.23): `fluxlib/gapfill/ggapfill.py` (`GFiller`),
`fluxlib/gapfill/utils.py` (`make_gap_pipeline`, `make_gaps`),
`fluxlib/gapfill/dataloader.py`, `fluxlib/gapfill/config-examples/*.yaml`, and
the notebook checkpoints `fluxnet_artificial_gapfill`, `artificial_gapfill`,
`rfr_gapfill_fluxnet`, `gapfill_fluxnet` and `analyze_result_of_artificial_gaps`.
The archived `fluxlib-0.0.29` snapshot differs from `da53256` in two cosmetic lines
(an import path and `np.int` -> `int`).

**Limitation.** The repository does not contain the scripts that produced the
article's tables. What follows is the behaviour of the library the paper cites,
corroborated where possible by the archived notebooks; it is evidence about the
published method, not a transcript of the published runs.

Every behavioural claim about `set_stats` below was confirmed by running a
transcription of it (`tests/test_legacy.py::fluxlib_set_stats`).

## 2. Classification legend

| Class | Meaning |
|---|---|
| **paper-consistent** | `fluxlib` does what the article states, or what this package already chose |
| **paper-ambiguous** | the article is silent; `fluxlib` supplies one answer |
| **paper-conflicting** | `fluxlib` does something other than what the article states |
| **later repository change** | behaviour introduced after publication |
| **not stated** | an implementation detail the article does not address and that has no counterpart decision in the specification |

## 3. Findings

### 3.1 Receptive-limiter features

| # | Topic | Article / spec | `fluxlib` 0.0.23 | Class | Action here |
|---|---|---|---|---|---|
| F1 | When the daily statistics are computed (A6) | from QC-controlled target observations; silent on timing relative to the holdout | `set_stats` runs on the whole frame inside `run_filling_pipeline`, **before** `itrain`/`itest` are applied; the notebook `run_filling` does the same. Held-out values are in the frame, so they shape their own days' statistics. | paper-ambiguous, resolved by evidence in the leaky direction | reproduced **only** in `legacy_fluxlib`, with `LegacyFluxlibWarning`; `paper_safe` unchanged |
| F2 | Which statistics | Q1, Q2, Q3, standard deviation | max, min, mean, std, p25, p50, p75 (seven) | paper-conflicting | `legacy_fluxlib` only |
| F3 | Days without observations (A4) | silent | the target is linearly interpolated over row position (`Series.interpolate()`) across every gap before the statistics are taken, so no day inside a gap lacks statistics; leading days before the first value do | paper-ambiguous | `legacy_fluxlib` only; `paper_safe` keeps `missing` and the three reaching strategies |
| F4 | Joining a day's statistics back to rows | "joined back to every timestamp of that day" | `daily.resample("15T").bfill()`: the row at exactly midnight gets its own day's value, **every other row gets the next day's**; rows after the last midnight repeat the last day's (closing `df.interpolate()`). Present in every release, including the 0.0.13 `resample("30T").bfill()` | paper-conflicting (almost certainly unintended) | reproduced exactly in `legacy_fluxlib` |
| F5 | Standard deviation and quantile convention (A11) | silent | pandas defaults: `ddof=1`, linear quantiles | paper-consistent with this package's A11 default | none; A11 default confirmed |
| F6 | Time feature | hours elapsed since the start of the series | `set_hour_diff` exists from 2020-07 onwards but `run_filling_pipeline` never calls it; the pipeline uses **day of year and year** | paper-conflicting | `legacy_fluxlib` uses `legacy_doy`, `legacy_year`; `paper_safe` keeps `time_distance_hours` |
| F7 | Season | hemisphere-aware season | up to 0.0.22 `(month % 12 + 3) // 3` (northern only); 0.0.23 adds `isnorth=False` -> `((month + 6) % 12 + 3) // 3`. Codes 1-4 = winter, spring, summer, autumn in either hemisphere - the same month groups and order as `season_tag`, offset by one. `isnorth` defaults to `True`. | paper-consistent (0.0.23); the fix itself is a repository change inside the paper period | reproduced; the hemisphere is still required explicitly, never defaulted to north |
| F8 | Radiation category boundaries (A2) | < 10, 10-100, > 100 W m-2 | strict comparisons with `np.select(..., default=0)`: a value **exactly** 10 or 100 matches nothing and becomes class 0, and so does a missing value | paper-ambiguous (neither of this package's two conventions) | `legacy_rg_rank` in `legacy_fluxlib`; `paper_safe` keeps `medium_inclusive` and "missing stays missing" |
| F9 | Drivers | pre-filled FLUXNET drivers | the closing `df.interpolate()` of `set_stats` also interpolates the drivers; prediction rows are `.interpolate().bfill()`-ed | not stated; a no-op on pre-filled drivers | **not reproduced**: it fabricates predictors, which `method_spec.md` section 7 forbids in every mode |

### 3.2 Model

| # | Topic | Article / spec | `fluxlib` 0.0.23 | Class | Action here |
|---|---|---|---|---|---|
| F10 | Hyperparameter search (A1) | tuned with `GridSearchCV`; grid not given | `GFiller.auto_optimize`: `bootstrap [True]`, `max_depth [80, 90, 100, 110]`, `max_features [2, 3]`, `min_samples_leaf [3, 4, 5]`, `min_samples_split [8, 10, 12]`, `n_estimators [100, 200, 300, 1000]`, `cv=3`. Defined in every release 0.0.13-0.0.29; **called by no archived pipeline or notebook** | paper-ambiguous: the grid exists as the paper describes, but nothing archived shows it ran | preset `hyperparameter_preset="legacy_fluxlib"` |
| F11 | Parameters actually fitted | - | every pipeline builds `RandomForestRegressor(max_depth=20, min_samples_leaf=3, min_samples_split=12, n_estimators=100)` from `ggapfill.yaml` (notebooks: `train_rfr(..., n_estimators=100)`); `max_features` at scikit-learn's default (all features) | paper-conflicting if these were the paper's runs; unverifiable | preset `hyperparameter_preset="legacy_fluxlib_fixed"` (a one-point grid) |
| F12 | Cross-validation inside the search (A5) | silent | `cv=3`: unshuffled 3-fold `KFold` for a regressor | paper-consistent with this package's `kfold` default; fold count differs (5 here) | none; pass `cv_folds=3` to repeat it |
| F13 | Seeding | - | `np.random.seed(seed)` globally; `GFiller` configs set no `random_state` | not stated | none; this package seeds every component explicitly |

### 3.3 Artificial-gap scenario

| # | Topic | Article / spec | `fluxlib` 0.0.23 | Class | Action here |
|---|---|---|---|---|---|
| F14 | Meaning of the 20/30/50 mix (A3) | ambiguous | `n_gap = int(len(series) * 0.25 * share / window)`: the shares apportion **records**, and the denominator is every row of the series, including real and QC-rejected gaps | paper-ambiguous; evidence supports the `missing_records` default | none; default confirmed. This package measures against genuinely observed values rather than all rows, which is documented, not changed |
| F15 | Gap lengths | 24 h, 7 d, 30 d | 48, 336, 1440 rows | paper-consistent at 30-minute cadence | none; elapsed time here |
| F16 | Placement order | - | 30-day gaps first, then 7-day, then 24-hour | paper-consistent with this package (longest first) | none |
| F17 | Minimum observed fraction | 50% | `vtheta=0.5` on non-missing values after the notebook sets QC != 0 to missing | paper-consistent | none |
| F18 | Overlap, boundaries, retries | - | a pool of 3x the needed start points; a candidate is dropped if within one window of *any* earlier drawn start (including rejected ones); a shortfall is silent | not stated | none; this package reports shortfalls (`GapError`, `GapScenarioWarning`) |
| F19 | Identical gaps for NEE, H, LE | yes | `make_gap_pipeline` accepts several fluxes (joint validity) and is seeded with 0, but the archived callers pass one flux at a time | paper-ambiguous in the archive | none; joint placement here |
| F20 | Train/test sets | withheld = test, rest = train (Figure 2) | train = non-gap rows with a value, test = gap rows with a value; a random 67/33 split only when no indices are passed | paper-consistent | none |

### 3.4 Metrics

| # | Topic | Article / spec | `fluxlib` 0.0.23 | Class | Action here |
|---|---|---|---|---|---|
| F21 | R2 (A12) | "coefficient of determination", beside a slope | `scipy.stats.linregress(truth, estimates).rvalue ** 2` - the **squared Pearson correlation**, in `GFiller.test` and in the result-analysis notebook | paper-ambiguous; evidence points to `squared_correlation` | default `residual` kept (section 5); set `r2_definition="squared_correlation"` for a historical comparison |
| F22 | Slope, RMSE, bias | slope of filled on measured; `(sum(filled) - sum(measured)) / n` | `linregress(truth, estimates)` slope; `sqrt(mean_squared_error)`; `mean(estimates - truth)` | paper-consistent | none |
| F23 | Day/night split, energy-balance ratio | daytime `SW_IN > 20`; EBR | not present in the archived gap-filling code | no evidence | none |

### 3.5 Other observations

| # | Topic | Observation | Class | Action |
|---|---|---|---|---|
| F24 | Notebook driver configuration | the archived `rfr_gapfill_fluxnet_cfg.yaml` checkpoint uses ERA drivers (`TA_ERA`, `SW_IN_ERA`, `VPD_ERA`) and lists `NEE_VUT_REF_QC` among the **drivers** | paper-conflicting if it described the paper's runs; it is an exploratory checkpoint on the pre-0.0.14 API and is not treated as evidence of them | none |
| F25 | Post-publication changes | 0.0.26 adds `gapfill_routine.py` (other regressors, a "last one third" split, `vtheta=0.1`, `rbak=10`); 2024 edits; in 2025 the module is removed in favour of a bootstrapped XGBoost ensemble with a 0.66 valid-ratio rule | later repository change | none |

## 4. What was implemented

**`feature_mode="legacy_fluxlib"`** (`rfrgapfill.legacy`). Reproduces F1-F4 and
F6-F8 exactly, under its own column names so a model fitted in one mode rejects a
matrix from the other:

```text
<drivers>, <target>_legacy_max, _legacy_min, _legacy_mean, _legacy_std,
_legacy_p25, _legacy_p50, _legacy_p75, legacy_season, legacy_rg_rank,
legacy_doy, legacy_year
```

- `build_validation_features` computes the legacy statistics from every
  *observed* value, held-out ones included, and emits `LegacyFluxlibWarning` on
  every call. Training rows and scored rows are unchanged.
- `detect_target_leakage` reports exactly the seven legacy statistics, and
  `require_no_target_leakage` raises with an explanation naming this mode.
- The settings `fluxlib`'s rules replace (`radiation_thresholds`,
  `boundary_convention`, `min_daily_observations`, `daily_statistic_strategy`,
  `fallback_window_days`, `daily_std_ddof`) cannot be changed in this mode, and
  the manifest reports them as `null`; `ambiguity_choices()` records `fluxlib`'s
  rules under A2 and A4 instead.
- `is_paper_faithful` is `False`, and RFR arms are labelled `RFR3-legacy` /
  `RFR10-legacy`, so no table, median across sites or benchmark comparison can
  pool them with a `paper_safe` arm. ORF builds no target-derived feature, so it
  keeps its usual label.

Not reproduced, deliberately: F9 (driver interpolation and prediction-row
back-filling), F13 (global seeding), and `fluxlib`'s driver column order.
Bit-for-bit agreement with 2021 forests is not a goal in any case - scikit-learn
has changed underneath them.

**Hyperparameter presets** (`RFRConfig.hyperparameter_preset`).

| Preset | Grid |
|---|---|
| `package_default` | `DEFAULT_HYPERPARAMETER_GRID`, unchanged |
| `legacy_fluxlib` | F10, verbatim (288 candidates; pair with `cv_folds=3`) |
| `legacy_fluxlib_fixed` | F11, one point |

A preset and an explicit grid must agree, `with_hyperparameter_preset()` switches
preset on an existing configuration, and a grid equal to a preset is recognised as
that preset in the manifest.

## 5. What was not changed, and why

The step's rule is not to replace `paper_safe` defaults merely to match old
source code. Each default below stays because the article either states
otherwise or leaves the choice open and the specification's reason for it still
holds:

- **Leakage-safe daily statistics (A6)** - the artificial-gap experiment is
  meant to measure skill on unseen data; F1 would defeat it.
- **Four statistics, joined to their own day, elapsed-hours time feature** - F2,
  F4 and F6 conflict with the article's own description.
- **`daily_statistic_strategy="missing"` (A4)** - interpolating the target across
  a gap (F3) invents target values; the reaching strategies stay available.
- **`medium_inclusive` radiation bins (A2)** - F8's "exactly on a threshold or
  missing -> 0" merges unrelated cases into one class.
- **`r2_definition="residual"` (A12)** - F21 is good evidence for what the paper
  reported, but the squared correlation is blind to systematic offsets; both
  remain implemented and every result records which it used. Section 6.1 of
  `method_spec.md` notes the two coincide in the regime Table S3 sits in.
- **Package default grid (A1)** - F10 was never demonstrably run and costs 288
  candidates per fit.

## 6. Running a paired comparison

The useful question the legacy mode answers is how much of the historical skill
comes from F1. Run both arms on the same gaps:

```python
from rfrgapfill import FeatureConfig, validate_rfr

safe = validate_rfr(df, config=config, targets="NEE", qc_columns="NEE_VUT_REF_QC")

historical = config.replace(
    features=FeatureConfig(feature_mode="legacy_fluxlib"),
    cv_folds=3,
).with_hyperparameter_preset("legacy_fluxlib_fixed")

legacy = validate_rfr(
    df, config=historical, targets="NEE", qc_columns="NEE_VUT_REF_QC", gaps=safe.gaps
)
```

`safe` is labelled `RFR3` and `legacy` `RFR3-legacy` in every table. Add
`validation=ValidationConfig(r2_definition="squared_correlation")` to the
historical configuration to score it as `fluxlib` did (F21).

## 7. Evidence index

| Finding | Location at `da53256` unless stated |
|---|---|
| F1, F9 | `fluxlib/gapfill/ggapfill.py`: `run_filling_pipeline` (`set_stats` precedes the `itrain`/`itest` split; `X_apply ... .interpolate().bfill()`); notebook `fluxnet_artificial_gapfill`, cell 2 `run_filling` |
| F2-F5 | `ggapfill.py`: `GFiller.set_stats` |
| F6 | `ggapfill.py`: `set_hour_diff` (defined, uncalled), `set_doy_year_tag`; `git log -S set_hour_diff` first hit `23fd77a` (2020-07-10) |
| F7 | `ggapfill.py`: `set_season_tag`; `isnorth` first appears in `da53256` |
| F8 | `ggapfill.py`: `set_rg_tag` |
| F10 | `ggapfill.py`: `auto_optimize`; `git grep auto_optimize` across all commits finds only definitions |
| F11 | `gapfill/config-examples/ggapfill.yaml`; 0.0.13 `gapfill_rfr.py` `train_rfr` defaults; notebook calls `train_rfr(..., n_estimators=100)` |
| F14-F19 | `fluxlib/gapfill/utils.py`: `make_gap_pipeline`, `make_gaps`; `config-examples/make_gaps_cfg.yaml`; notebook `fluxnet_artificial_gapfill`, cells 2-3 |
| F20 | `ggapfill.py`: `run_filling_pipeline` |
| F21, F22 | `ggapfill.py`: `GFiller.test`; notebook `analyze_result_of_artificial_gaps` |
| F24 | `.ipynb_checkpoints/rfr_gapfill_fluxnet_cfg-checkpoint.yaml` |
| F25 | `fluxlib-history-versions/fluxlib-0.0.29/gapfill/gapfill_routine.py`; `fluxlib/gapfilling.py` at `c837be3` |
