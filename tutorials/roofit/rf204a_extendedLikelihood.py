## \file
## \ingroup tutorial_roofit
## \notebook
## Extended maximum likelihood fit in multiple ranges.
##
## \macro_code
##
## \date March 2021
## \authors Harshal Shende, Stephan Hageboeck (C++ version)

import ROOT


# Setup component pdfs
# ---------------------

# Declare observable x
x = ROOT.RooRealVar("x", "x", 0, 11)

# Create two Gaussian PDFs g1(x,mean1,sigma) anf g2(x,mean2,sigma) and their parameters
mean = ROOT.RooRealVar("mean", "mean of gaussians", 5)
sigma1 = ROOT.RooRealVar("sigma1", "width of gaussians", 0.5)
sigma2 = ROOT.RooRealVar("sigma2", "width of gaussians", 1)

sig1 = ROOT.RooGaussian("sig1", "Signal component 1", x, mean, sigma1)
sig2 = ROOT.RooGaussian("sig2", "Signal component 2", x, mean, sigma2)

# Build Chebychev polynomial pdf
a0 = ROOT.RooRealVar("a0", "a0", 0.5, 0.0, 1.0)
a1 = ROOT.RooRealVar("a1", "a1", 0.2, 0.0, 1.0)
bkg = ROOT.RooChebychev("bkg", "Background", x, ROOT.RooArgSet(a0, a1))

# Sum the signal components into a composite signal pdf
sig1frac = ROOT.RooRealVar("sig1frac", "fraction of component 1 in signal", 0.8, 0.0, 1.0)
sig = ROOT.RooAddPdf("sig", "Signal", ROOT.RooArgList(sig1, sig2), sig1frac)


# Extend the pdfs
# -----------------------------

# Define signal range in which events counts are to be defined
x.setRange("signalRange", 4, 6)

# Associated nsig/nbkg as expected number of events with sig/bkg _in_the_range_ "signalRange"
nsig = ROOT.RooRealVar("nsig", "number of signal events in signalRange", 500, 0.0, 10000)
nbkg = ROOT.RooRealVar("nbkg", "number of background events in signalRange", 500, 0, 10000)

# Use AddPdf to extend the model. Giving as many coefficients as pdfs switches on extension.
model = ROOT.RooAddPdf("model", "(g1+g2)+a", ROOT.RooArgList(bkg, sig), ROOT.RooArgList(nbkg, nsig))

# Sample data, fit model
# -------------------------------------------

# Generate 1000 events from model so that nsig,nbkg come out to numbers <<500 in fit
data = model.generate(x, 1000)

canv = ROOT.TCanvas("Canvas", "Canvas", 1500, 600)
canv.Divide(3, 1)

# Fit full range
#  -------------------------------------------

# Perform unbinned ML fit to data, full range

# IMPORTANT:
# The model needs to be copied when fitting with different ranges because
# the interpretation of the coefficients is tied to the fit range
# that's used in the first fit
canv.cd(1)

model1 = ROOT.RooAddPdf(model)
r = model1.fitTo(data, Save=True)
r.Print()

frame = x.frame(ROOT.RooFit.Title("Full range fitted"))
data.plotOn(frame)
model1.plotOn(frame, VisualizeError=r)
model1.plotOn(frame)
model1.paramOn(frame)
frame.Draw()


# Fit in two regions
# -------------------------------------------

canv.cd(2)
x.setRange("left", 0.0, 4.0)
x.setRange("right", 6.0, 10.0)

model2 = ROOT.RooAddPdf(model)
r2 = model2.fitTo(data, Range="left,right", Save=True)
r2.Print()

frame2 = x.frame(ROOT.RooFit.Title("Fit in left/right sideband"))
data.plotOn(frame2)
model2.plotOn(frame2, VisualizeError=r2)
model2.plotOn(frame2)
model2.paramOn(frame2)
frame2.Draw()


# Fit in one region
# -------------------------------------------
# Note how restricting the region to only the left tail increases
# the fit uncertainty

canv.cd(3)
x.setRange("leftToMiddle", 0.0, 5.0)

model3 = ROOT.RooAddPdf(model)
r3 = model3.fitTo(data, Range="leftToMiddle", Save=True)
r3.Print()

frame3 = x.frame(ROOT.RooFit.Title("Fit from left to middle"))
data.plotOn(frame3)
model3.plotOn(frame3, VisualizeError=r3)
model3.plotOn(frame3)
model3.paramOn(frame3)
frame3.Draw()

canv.Draw()

canv.SaveAs("rf204a_extendedLikelihood.png")
