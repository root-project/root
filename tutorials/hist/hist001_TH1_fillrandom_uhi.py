## \file hist001_TH1_first.py
### \ingroup tutorial_hist
## \notebook
## Fill a 1D histogram with random values using predefined functions.
##
## \macro_image
## \macro_code
##
## \date July 2025
## \author Giacomo Parolini, Nursena Bitirgen

import numpy as np
import ROOT

# Create a one dimensional histogram and fill it with a gaussian distribution
histogram = ROOT.TH1D("h1d", "Test random numbers", 200, 0.0, 10.0)
# "gaus" is a predefined ROOT function. Here we are filling the histogram with
# 10000 values sampled from that distribution.
values = np.random.normal(0.0, 1.0, 10000)
histogram[:] = np.histogram(values, bins=200, range=(0.0, 10.0))[0]
# Open a ROOT file and save the histogram
with ROOT.TFile.Open("fillrandom_py.root", "RECREATE") as myfile:
    myfile.WriteObject(histogram, histogram.GetName())
