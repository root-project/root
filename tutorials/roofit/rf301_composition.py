## \file
## \ingroup tutorial_roofit
## \notebook
## Multidimensional models: multi-dimensional pdfs through composition, e.g. substituting
## a pdf parameter with a function that depends on other observables
##
## `pdf = gauss(x,f(y),s)` with `f(y) = a0 + a1*y`
##
## \macro_code
##
## \date February 2018
## \authors Clemens Lange, Wouter Verkerke (C++ version)

import ROOT

# Setup composed model gauss(x, m(y), s)
# -----------------------------------------------------------------------

# Create observables
x = ROOT.RooRealVar("x", "x", -5, 5)
y = ROOT.RooRealVar("y", "y", -5, 5)

# Create function f(y) = a0 + a1*y
a0 = ROOT.RooRealVar("a0", "a0", -0.5, -5, 5)
a1 = ROOT.RooRealVar("a1", "a1", -0.5, -1, 1)
fy = ROOT.RooPolyVar("fy", "fy", y, [a0, a1])

# Creat gauss(x,f(y),s)
sigma = ROOT.RooRealVar("sigma", "width of gaussian", 0.5)
model = ROOT.RooGaussian("model", "Gaussian with shifting mean", x, fy, sigma)

# Sample data, plot data and pdf on x and y
# ---------------------------------------------------------------------------------

# Generate 10000 events in x and y from model
data = model.generate({x, y}, 10000)

# Plot x distribution of data and projection of model x = Int(dy)
# model(x,y)
xframe = x.frame()
data.plotOn(xframe)
model.plotOn(xframe)

# Plot x distribution of data and projection of model y = Int(dx)
# model(x,y)
yframe = y.frame()
data.plotOn(yframe)
model.plotOn(yframe)

# Make two-dimensional plot in x vs y
hh_model = model.createHistogram("hh_model", x, Binning=50, YVar=dict(var=y, Binning=50))
hh_model.SetLineColor(ROOT.kBlue)

# Make canvas and draw ROOT.RooPlots
c = ROOT.TCanvas("rf301_composition", "rf301_composition", 1200, 400)
c.Divide(3)
c.cd(1)
ROOT.gPad.SetLeftMargin(0.15)
xframe.GetYaxis().SetTitleOffset(1.4)
xframe.Draw()
c.cd(2)
ROOT.gPad.SetLeftMargin(0.15)
yframe.GetYaxis().SetTitleOffset(1.4)
yframe.Draw()
c.cd(3)
ROOT.gPad.SetLeftMargin(0.20)
hh_model.GetZaxis().SetTitleOffset(2.5)
hh_model.Draw("surf")

c.SaveAs("rf301_composition.png")
