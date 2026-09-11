# Command-line reference

`rfr-gapfill` is installed with the package. It is a thin shell over the Python
API: `validate` runs `validate_rfr` and `fill` runs `RFRGapFiller`, with the
configuration file read by `load_config`. The README's
[command-line section](../README.md#command-line) explains the configuration file
and what each subcommand writes; this page lists every option, generated from the
parser itself.

```{eval-rst}
.. argparse::
   :module: rfrgapfill.cli
   :func: build_parser
   :prog: rfr-gapfill
```
