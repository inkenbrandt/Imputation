# Test fixtures

Small, hand-checkable inputs and expected outputs used by the unit tests.
Fixtures whose expected values come from the journal supplements must cite the
supplementary table or figure they came from, per `docs/supplement_benchmarks.md`.
Real FLUXNET2015 data is never committed here.

Anything larger than a hand-checkable table is generated rather than stored.
`rfrgapfill.synthetic.synthetic_site()` builds a deterministic year of
half-hourly data with known diurnal, seasonal and flux structure, and
`known_gaps()` places fixed 24-hour, 7-day and 30-day intervals over it — so a
scientific fixture is a seed and a function call, not a committed file, and no
test depends on downloading anything. See `tests/test_synthetic.py` for what the
generator guarantees.
