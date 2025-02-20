## \file
## \ingroup tutorial_hist
## \notebook
## Fill a 1D histogram with random values using predefined functions.
##
## \macro_image
## \macro_code
##
## \date November 2024
## \author Giacomo Parolini
import ROOT

# Create a one dimensional histogram and fill it with a gaussian distribution
h1d = ROOT.TH1D("h1d", "Test random numbers", nbinsx = 200, xlow = 0.0, xup = 10.0)

# "gaus" is a predefined ROOT function. Here we are filling the histogram with
# 10000 values sampled from that distribution.
h1d.FillRandom("gaus", 10000)

# Open a ROOT file and save the histogram
with ROOT.TFile.Open("fillrandom_py.root", "RECREATE") as myfile:
   myfile.WriteObject(h1d, h1d.GetName())
