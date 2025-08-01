## \file
## \ingroup tutorial_hist
## \notebook
## Example creating a simple ratio plot of two histograms using the `pois` division option.
## Two histograms are set up and filled with random numbers. The constructor of `TRatioPlot`
## takes the to histograms, name and title for the object, drawing options for the histograms (`hist` and `E` in this case)
## and a drawing option for the output graph.
## Inspired by the tutorial of Paul Gessinger.
##
## \macro_image
## \macro_code
## \date July 2025
## \author Alberto Ferro, Nursena Bitirgen

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import ROOT

ROOT.gStyle.SetOptStat(0)
c1 = ROOT.TCanvas("c1", "A ratio example")
h1 = ROOT.TH1D("h1", "h1", 50, 0, 10)
h2 = ROOT.TH1D("h2", "h2", 50, 0, 10)
f1 = ROOT.TF1("f1", "exp(- x/[0] )")
f1.SetParameter(0, 3)
h1[...] = np.histogram(np.array([f1.GetRandom() for _ in range(1900)]), bins=50, range=(0.0, 10.0))[0]
h2[...] = np.histogram(np.array([f1.GetRandom() for _ in range(2000)]), bins=50, range=(0.0, 10.0))[0]
hep.histplot(h1, h2)

plt.show()
