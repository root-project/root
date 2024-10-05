# RooFit pythonizations

## How to work on the RooFit pythonizations

The RooFit pythonizations come installed with ROOT if ROOT was built with RooFit enabled.

However, the `rofit_pythonization` package can also be installed standalone with pip, in which case it takes priority over the pythonizations inside ROOT.

Therefore, the easiest and recommended way to develop RooFit pythonizations is as follows:

  1. Make sure ROOT and RooFit are installed on your system
  2. Clone the ROOT repository and go inside the `roofit/pythonizations` directory
  3. Install with `pip install -e .` for an editable install (virtual environment recommended)
  4. Any changes to the RooFit pythonizations will be in effect immediately without recompiling or installing anything!
  5. Make your changes, open PR to ROOT repository

And please remember to add tests in the `test` subdirectory.
