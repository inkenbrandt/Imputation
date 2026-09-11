# Examples

- `synthetic_example.py`: an end-to-end demonstration on synthetic half-hourly
  data with known diurnal and seasonal structure (Step 17, acceptance tests
  34-35). It runs RFR3 and RFR10 over NEE, H and LE on one shared set of
  24 h / 7 d / 30 d artificial gaps and prints the metric tables, the gap
  manifest, the bias spread by gap class, the energy-balance comparison and the
  two arms against gap duration (Step 18). Then it checks that no observed value
  changed. The published Table S3 medians are printed for reference only; the
  synthetic site is not a FLUXNET reproduction and nothing is compared against
  them. `--full-grid` searches the package's default hyperparameter grid, and
  `--manifests DIR` writes one run manifest per target.
- `notebooks/`: four Jupyter notebooks, committed with their outputs, and
  described in [`docs/examples.md`](../docs/examples.md):
  - [`01_getting_started.ipynb`](notebooks/01_getting_started.ipynb): a first
    fill, its provenance columns and its run manifest;
  - [`02_artificial_gap_validation.ipynb`](notebooks/02_artificial_gap_validation.ipynb):
    the gap scenario, long gaps (A4), leakage and `validate_rfr`;
  - [`03_comparing_configurations.ipynb`](notebooks/03_comparing_configurations.ipynb):
    RFR3, RFR10 and ORF, gap-length figures and the published medians;
  - [`04_bring_your_own_data.ipynb`](notebooks/04_bring_your_own_data.ipynb): a
    FLUXNET2015-format file, a hand-built configuration and the command line.

  Notebooks are an optional extra (`pip install -e ".[notebooks]"`). The
  package core and its default test run never require them.
