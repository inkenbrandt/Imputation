# Examples

- `synthetic_example.py` — end-to-end demonstration on synthetic half-hourly data
  with known diurnal and seasonal structure (Step 17, acceptance tests 34-35).
  Runs RFR3 and RFR10 over NEE, H and LE on one shared set of 24 h / 7 d / 30 d
  artificial gaps and prints the metric tables, the gap manifest, the bias spread
  by gap class, the energy-balance comparison and the two arms lined up against
  gap duration (Step 18), then checks that no observed value changed. The
  published Table S3 medians are printed for reference only; the synthetic site
  is not a FLUXNET reproduction and nothing is compared against them.
  `--full-grid` searches the package's default hyperparameter grid;
  `--manifests DIR` writes one run manifest per target.
- A FLUXNET notebook (`fluxnet_example.ipynb`) is planned. Notebooks are an
  optional extra (`pip install -e ".[notebooks]"`) and are never required by the
  package core or its tests.
