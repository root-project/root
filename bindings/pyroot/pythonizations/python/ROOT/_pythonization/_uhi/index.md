\defgroup Py_UHI Unified Histogram Interface (UHI)
\ingroup Python
\brief Using ROOT histograms in Python

# ROOT Histograms & UHI

ROOT histograms implement the [Unified Histogram Interface (UHI)](https://uhi.readthedocs.io/en/latest/index.html), a standard protocol that makes ROOT histograms interoperable with the broader Python
scientific ecosystem. This compliance standardizes histogram operations, making tasks like plotting, indexing, and slicing more intuitive and consistent.

\note UHI support is available for all [`TH1`](https://root.cern.ch/doc/master/classTH1.html)-derived
classes, including [`TH2`](https://root.cern.ch/doc/master/classTH2.html) and
[`TH3`](https://root.cern.ch/doc/master/classTH3.html).

## Cheat Sheet

A one-page quick reference covering the API.

\htmlonly
<object data="uhi-cheatsheet.pdf"
        type="application/pdf"
        width="100%"
        height="520px"
        style="border:1px solid #ccc;border-radius:6px;">
  <p>PDF preview not available in your browser.</p>
</object>
<a href="uhi-cheatsheet.pdf"
   style="display:inline-block;margin-top:8px;padding:6px 14px;background:#1a73e8;
          color:#fff;border-radius:4px;text-decoration:none;font-size:13px;">
  ⬇ Download cheat sheet (PDF)
</a>
\endhtmlonly


\anchor plotting
# Plotting

ROOT histograms implement the `PlottableHistogram` protocol. Any plotting library that accepts an object that follows the protocol can plot ROOT histogram objects.

You can read more about the protocol on the [UHI plotting](https://uhi.readthedocs.io/en/latest/plotting.html) page.

\anchor plotting-with-mplhep
## Plotting with [mplhep](https://github.com/scikit-hep/mplhep)
```python
import ROOT
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

# Create and fill a 1D histogram 
h1 = ROOT.TH1D("h1", "MyHist", 10, -1, 1)
arr = np.random.normal(0, 1, 1000)
h1.Fill(arr)

# Load a style sheet and plot the histogram
hep.style.use("LHCb2")
hep.histplot(h1)
plt.title("MyHist")
plt.show()
```

For 2D histograms, use `hep.hist2dplot`:

```python
h2 = ROOT.TH2D("h2", "h2", 10, -1, 1, 10, -1, 1)
h2.FillRandom("gaus", 10000)

hep.hist2dplot(h2)
plt.show()
```

\anchor indexing
# Indexing

ROOT histograms implement the UHI indexing specification. This introduces a unified syntax for accessing and setting bin values, as well as slicing histogram axes.

You can read more about the syntax on the [UHI Indexing](https://uhi.readthedocs.io/en/latest/indexing.html) page.

\anchor setup
## Setup
The `loc`, `undeflow`, `overflow`, `rebin` and `sum` tags are imported from the `ROOT.uhi` module.
```python
import ROOT
from ROOT.uhi import loc, underflow, overflow, rebin, sum
import numpy as np


h = ROOT.TH2D("h2", "h2", 10, 0, 1, 10, 0, 1)
```

\anchor slicing
## Slicing
```python
# Slicing over everything
h == h[:, :]
h == h[...]

# Slicing a range, picking the bins 1 to 5 along the x axis and 2 to 6 along the y axis
h1 = h[1:5, 2:6]

# Slicing leaving out endpoints
h2 = h[:5, 6:]

# Slicing using data coordinates, picking the bins from the one containing the value 0.5 onwards along both axes
h3 = h[loc(0.5):, loc(0.5):]

# Combining slicing with rebinning, rebinning the x axis by a factor of 2
h4 = h[1:9:rebin(2), :]
```

\anchor setting
### Setting
```python
# Setting the bin contents
h[1, 2] = 5

# Setting the bin contents using data coordinates
h[loc(3), loc(1)] = 5

# Setting the flow bins
h[overflow, overflow] = 5

# Setting the bin contents using a numpy array
h[...] = np.ones((10, 10))

# Setting the bin contents using a scalar
h[...] = 5
```

\anchor access
### Access
```python
# Accessing the bin contents using the bin number
v = h[1, 2]

# Accessing the bin contents using data coordinates
v = h[loc(0.5), loc(0.5)]
v = h[loc(0.5) + 1, loc(0.5) + 1] # Returns the bin above the one containing the value 2 along both axes

# Accessing the flow bins
v = h[underflow, underflow]
```

\anchor additional-notes-2
## Additional Notes

- **Indexing system**
        - ROOT histograms use a bin indexing system that ranges from 0 to `nbins+1` where 0 is the underflow bin and `nbins+1` is the overflow (see [conventions for numbering bins](https://root.cern.ch/doc/master/classTH1.html#convention)). In contrast, UHI inherits 0-based indexing from numpy array conventions where 0 is the first valid element and n-1 is the last valid element. Our implementation complies with the UHI conventions by implementing the following syntax:
                - `h[underflow]` returns the underflow bin (equivalent to `h.GetBinContent(0)`).
                - `h[0]` returns the first valid bin (equivalent to `h.GetBinContent(1)`).
                - `h[-1]` returns the last valid bin (equivalent to `h.GetBinContent(nbins)`).
                - `h[overflow]` returns the overflow bin (equivalent to `h.GetBinContent(nbins+1)`).
- **Slicing operations** 
        - Slicing always returns a new histogram with the appropriate values copied from the original one according to the input slice.
        - Values outside of the slice range fall into the flow bins.
- **Summing operations with `sum`**
        - For a 1D histogram, the integral of the selected slice is returned.
                - ex. `ans = h[a:b:sum]` --> `ans` is the integral value.
        - For a 2D or 3D histogram, a new histogram with reduced dimensionality is returned 
                - ex. `h_projected = h[:, ::sum, ::sum]` --> `h_projected` is a 1D histogram representing the y and z projections along the x axis.
- **Setting operations**
        - Setting with a scalar does not set the flow bins.
        - Setting with an array checks whether the array matches the shape of the histogram with flow bins or the size without flow bins.


# Serialization

ROOT histograms can be serialized to a [shared UHI format](https://uhi.readthedocs.io/en/latest/serialization.html)
and deserialized into any UHI-compatible library, enabling histogram exchange between ROOT, boost-histogram, hist and others without manual conversion.

```python
import json, uhi.io.json, hist

# ROOT histogram → JSON
ob = json.dumps(h_root, default=uhi.io.json.default)

# JSON → any UHI-compatible library
ir     = json.loads(ob, object_hook=uhi.io.json.object_hook)
h_hist = hist.Hist(ir)
```
