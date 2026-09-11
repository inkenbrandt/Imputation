# API reference

One page per module, grouped in the order a run uses them. Most names are also
exported from the top-level `rfrgapfill` package, which is how the guides and
notebooks import them:

```python
from rfrgapfill import RFRConfig, RFRGapFiller, validate_rfr
```

The module docstrings carry the reasoning behind each layer; the
[method](../method.md) and [validation](../validation.md) guides are the gentler
introduction.

## Configuration

```{eval-rst}
.. autosummary::
   :toctree: generated

   rfrgapfill.config
   rfrgapfill.schema
```

## Input data

```{eval-rst}
.. autosummary::
   :toctree: generated

   rfrgapfill.time
   rfrgapfill.fluxnet
   rfrgapfill.synthetic
```

## Features

```{eval-rst}
.. autosummary::
   :toctree: generated

   rfrgapfill.features
   rfrgapfill.leakage
   rfrgapfill.legacy
```

## Model and filling

```{eval-rst}
.. autosummary::
   :toctree: generated

   rfrgapfill.model
   rfrgapfill.fill
```

## Validation

```{eval-rst}
.. autosummary::
   :toctree: generated

   rfrgapfill.gaps
   rfrgapfill.validation
   rfrgapfill.metrics
```

## Reporting and provenance

```{eval-rst}
.. autosummary::
   :toctree: generated

   rfrgapfill.sensitivity
   rfrgapfill.benchmarks
   rfrgapfill.uncertainty
   rfrgapfill.sites
   rfrgapfill.plotting
   rfrgapfill.provenance
```

## Command line

```{eval-rst}
.. autosummary::
   :toctree: generated

   rfrgapfill.cli
```

The options themselves are listed in the [command-line reference](../cli.md).
