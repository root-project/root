\defgroup Python Python Interface
\brief Python bindings and utilities for ROOT.

# Python Interface

ROOT is a C++ framework that exposes its full API to Python through dynamic bindings generated at runtime via [cppyy](https://cppyy.readthedocs.io/). Every ROOT class is available  in Python automatically without manual wrapping.
Additional pythonizations are layered on top of some classes to make the experience feel natively Pythonic.

## Installation

\htmlonly
<div class="install-tabs">
\endhtmlonly

**pip**
\code{.sh}
pip install root
\endcode

**conda**
\code{.sh}
conda install -c conda-forge root
\endcode

\htmlonly
</div>
\endhtmlonly


Installation with pip:
~~~{.sh}
pip install root
~~~

Installation with conda:
~~~{.sh}
conda install -c conda-forge root
~~~

## Quickstart

~~~{.sh}
import ROOT

df = ROOT.RDataFrame("tree", "file.root")
df.Histo1D("px").Draw()
~~~

## Topics

| Module | Description |
|---|---|
| @ref Py_RDataFrame "RDataFrame" | Analyse ROOT data with a high-level Python API |
| @ref Py_IO "I/O" | Reading and writing ROOT files, TTree and RNTuple from Python |
| @ref Py_UHI "UHI — Unified Histogram Interface" | Using ROOT histograms in Python |
| @ref Py_ML "ML / RDataLoader" | Feed ROOT data directly into models for training |
| @ref Py_Interop "Interoperability" | Using ROOT alongside other Python packages |
| @ref Pythonizations "Pythonizations" | Advanced — how ROOT adapts C++ classes for idiomatic Python use |
