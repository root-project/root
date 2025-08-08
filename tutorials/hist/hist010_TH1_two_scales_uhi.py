## \file
## \ingroup tutorial_hist
## \notebook
## Example of macro illustrating how to superimpose two histograms
## with different scales in the "same" pad.
## Inspired by work of Rene Brun.
##
## \macro_image
## \macro_code
##
## \date July 2025
## \author Alberto Ferro, Nursena Bitirgen

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import ROOT

np.random.seed(0)
plt.style.use(hep.style.ROOT)
fig, ax1 = plt.subplots(figsize=(10, 6))

h1 = ROOT.TH1F("h1", "my histogram", 100, -3, 3)

h1[...] = np.histogram(np.random.normal(0, 1, 10000), range=(-3, 3), bins=100)[0]

hep.histplot(h1, ax=ax1, histtype="fill", color="white", edgecolor="blue", linewidth=1.5, alpha=0.5)

hint1 = ROOT.TH1F("hint1", "h1 bins integral", 100, -3, 3)

hint1[...] = np.cumsum(h1.values())

ax2 = ax1.twinx()
hep.histplot(hint1, ax=ax2, histtype="errorbar", color="red", marker="+", markersize=3)

ax1.set_xlim(-3, 3)
plt.title("Histogram with Cumulative Sum")
plt.show()
