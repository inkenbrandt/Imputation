% The landing page is the project README, included rather than copied so the
% two cannot drift apart. conf.py retargets the README's links for this site.

```{include} ../README.md
```

```{toctree}
:hidden:
:caption: Getting started

self
examples
notebooks/01_getting_started
notebooks/02_artificial_gap_validation
notebooks/03_comparing_configurations
notebooks/04_bring_your_own_data
```

```{toctree}
:hidden:
:caption: User guide

method
validation
assumptions
fluxnet
fluxnet_adapter
```

```{toctree}
:hidden:
:caption: Reference

api/index
cli
method_spec
supplement_benchmarks
fluxlib_audit
```
