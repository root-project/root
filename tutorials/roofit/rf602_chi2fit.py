## \file
## \ingroup tutorial_roofit
## \notebook
##
## 'LIKELIHOOD AND MINIMIZATION' RooFit tutorial macro #602
##
## Setting up a chi^2 fit to a binned dataset
##
## \macro_code
##
## \date February 2018
## \author Clemens Lange
## \author Wouter Verkerke (C version)

from __future__ import print_function
import ROOT


# Set up model
# ---------------------

# Declare observable x
x = ROOT.RooRealVar("x", "x", 0, 10)

# Create two Gaussian PDFs g1(x,mean1,sigma) anf g2(x,mean2,sigma) and
# their parameters
mean = ROOT.RooRealVar("mean", "mean of gaussians", 5)
sigma1 = ROOT.RooRealVar("sigma1", "width of gaussians", 0.5)
sigma2 = ROOT.RooRealVar("sigma2", "width of gaussians", 1)

sig1 = ROOT.RooGaussian("sig1", "Signal component 1", x, mean, sigma1)
sig2 = ROOT.RooGaussian("sig2", "Signal component 2", x, mean, sigma2)

# Build Chebychev polynomial p.d.f.
a0 = ROOT.RooRealVar("a0", "a0", 0.5, 0.0, 1.0)
a1 = ROOT.RooRealVar("a1", "a1", 0.2, 0.0, 1.0)
bkg = ROOT.RooChebychev("bkg", "Background", x, ROOT.RooArgList(a0, a1))

# Sum the signal components into a composite signal p.d.f.
sig1frac = ROOT.RooRealVar("sig1frac", "fraction of component 1 in signal", 0.8, 0.0, 1.0)
sig = ROOT.RooAddPdf("sig", "Signal", ROOT.RooArgList(sig1, sig2), ROOT.RooArgList(sig1frac))

# Sum the composite signal and background
bkgfrac = ROOT.RooRealVar("bkgfrac", "fraction of background", 0.5, 0.0, 1.0)
model = ROOT.RooAddPdf("model", "g1+g2+a", ROOT.RooArgList(bkg, sig), ROOT.RooArgList(bkgfrac))

# Create biuned dataset
# -----------------------------------------

d = model.generate(ROOT.RooArgSet(x), 10000)
dh = d.binnedClone()

# Construct a chi^2 of the data and the model.
# When a p.d.f. is used in a chi^2 fit, probability density scaled
# by the number of events in the dataset to obtain the fit function
# If model is an extended p.d.f, expected number events is used
# instead of the observed number of events.
ll = ROOT.RooLinkedList()
model.chi2FitTo(dh, ll)

# NB: It is also possible to fit a ROOT.RooAbsReal function to a ROOT.RooDataHist
# using chi2FitTo().

# Note that entries with zero bins are _not_ allowed
# for a proper chi^2 calculation and will give error
# messages
dsmall = d.reduce(ROOT.RooFit.EventRange(1, 100))
dhsmall = dsmall.binnedClone()
chi2_lowstat = ROOT.RooChi2Var("chi2_lowstat", "chi2", model, dhsmall)
print(chi2_lowstat.getVal())
