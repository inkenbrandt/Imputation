# Example notebooks

Four Jupyter notebooks in [`examples/notebooks/`](../examples/notebooks/) walk
through the package from a first fill to a comparison with the published
numbers. Each runs from start to finish on the synthetic site, or on a file
written from it, so nothing is downloaded. They are committed with their outputs,
so they can be read on GitHub without running anything.

| Notebook | What it covers | Read it with |
|---|---|---|
| [`01_getting_started.ipynb`](../examples/notebooks/01_getting_started.ipynb) | the input a site needs and what its QC flags mean; a configuration; fitting RFR3 on LE and filling its gaps; the six provenance columns; checking the fill against the synthetic truth; the run manifest | [`method.md`](method.md) |
| [`02_artificial_gap_validation.ipynb`](../examples/notebooks/02_artificial_gap_validation.ipynb) | the paper's 24 h / 7 d / 30 d gap scenario and what it achieved; the two readings of 20/30/50 (A3); which withheld rows each daily-statistic strategy can predict (A4); the leakage probe; `validate_rfr` and how to read its tables; how much the historical `fluxlib` derivation flatters a score | [`validation.md`](validation.md) |
| [`03_comparing_configurations.ipynb`](../examples/notebooks/03_comparing_configurations.ipynb) | RFR3, RFR10 and both ORF benchmarks on one shared set of gaps; gap-length tables and figures; what the receptive limiter contributes; the published Table S3 medians, the NEE unit rule (A10) and the benchmark figure | [`validation.md`](validation.md#comparing-rfr3-rfr10-and-orf), [`supplement_benchmarks.md`](supplement_benchmarks.md) |
| [`04_bring_your_own_data.ipynb`](../examples/notebooks/04_bring_your_own_data.ipynb) | a FLUXNET2015-format file: `-9999`, integer timestamps, numbered soil sensors; the column map; the time axis; a hand-built configuration saved and reloaded; filling with and without `refill_pre_filled`; model persistence; the same fill from the command line, giving the same numbers | [`fluxnet.md`](fluxnet.md) |

Start with the first. The others stand alone, but they assume you have seen
how a configuration is built.

## Running them

The notebooks need the optional `notebooks` extra, which adds Jupyter and
matplotlib. The package itself never depends on it.

```bash
pip install -e ".[notebooks]"
jupyter lab examples/notebooks
```

Each notebook runs in a minute or two at most on a laptop. To rerun one and
replace its saved outputs:

```bash
jupyter nbconvert --to notebook --execute --inplace examples/notebooks/01_getting_started.ipynb
```

## What they choose, and why

Three settings recur in every notebook and are worth knowing before you copy
one:

- **A one-point hyperparameter grid** (`{"n_estimators": (50,)}`), so each run
  takes seconds. Drop `hyperparameter_grid` to search the package's default
  grid ([A1](assumptions.md#a1-the-hyperparameter-grid)).
- **`daily_statistic_strategy="rolling_available"`**, because under the default a
  gap covering whole days cannot be predicted
  ([A4](assumptions.md#a4-days-with-too-few-measured-values)). Notebook 02 shows
  why.
- **The synthetic site**, which is a software check, not evidence about the
  method. Its scores say nothing about how RFR does at a real tower, and none of
  them is compared with the published medians as if it could.

## Kept working

`tests/test_docs.py` checks every notebook on every test run: it is valid,
its links resolve, it imports nothing beyond the package, pandas, NumPy,
matplotlib and the standard library, and its saved outputs come from one clean
run from top to bottom. With the `notebooks` extra installed,
`pytest -m notebooks` also executes each one. CI does this on every pull
request, so a change to the package that breaks a notebook fails the build.
