## \file
## \ingroup tutorial_roofit
## \notebook
## Basic functionality: numerical 1st, and 3rd order derivatives w.r.t. observables and parameters
##
## ```
## pdf = gauss(x,m,s)
## ```
##
## \macro_code
##
## \date February 2018
## \authors Clemens Lange, Wouter Verkerke (C++ version)

import ROOT

# Set up model
# ---------------------

# Declare variables x,mean, with associated name, title, value and allowed
# range
x = ROOT.RooRealVar("x", "x", -10, 10)
mean = ROOT.RooRealVar("mean", "mean of gaussian", 1, -10, 10)
sigma = ROOT.RooRealVar("sigma", "width of gaussian", 1, 0.1, 10)

# Build gaussian pdf in terms of x, and sigma
gauss = ROOT.RooGaussian("gauss", "gaussian PDF", x, mean, sigma)

# Create and plot derivatives w.r.t. x
# ----------------------------------------------------------------------

# Derivative of normalized gauss(x) w.r.t. observable x
dgdx = gauss.derivative(x, 1)

# Second and third derivative of normalized gauss(x) w.r.t. observable x
d2gdx2 = gauss.derivative(x, 2)
d3gdx3 = gauss.derivative(x, 3)

# Construct plot frame in 'x'
xframe = x.frame(Title="d(Gauss)/dx")

# Plot gauss in frame (i.e. in x)
gauss.plotOn(xframe)

# Plot derivatives in same frame
dgdx.plotOn(xframe, LineColor="m")
d2gdx2.plotOn(xframe, LineColor="r")
d3gdx3.plotOn(xframe, LineColor="kOrange")

# Create and plot derivatives w.r.t. sigma
# ------------------------------------------------------------------------------

# Derivative of normalized gauss(x) w.r.t. parameter sigma
dgds = gauss.derivative(sigma, 1)

# Second and third derivative of normalized gauss(x) w.r.t. parameter sigma
d2gds2 = gauss.derivative(sigma, 2)
d3gds3 = gauss.derivative(sigma, 3)

# Construct plot frame in 'sigma'
sframe = sigma.frame(Title="d(Gauss)/d(sigma)", Range=(0.0, 2.0))

# Plot gauss in frame (i.e. in x)
gauss.plotOn(sframe)

# Plot derivatives in same frame
dgds.plotOn(sframe, LineColor="m")
d2gds2.plotOn(sframe, LineColor="r")
d3gds3.plotOn(sframe, LineColor="kOrange")

# Draw all frames on a canvas
c = ROOT.TCanvas("rf111_derivatives", "rf111_derivatives", 800, 400)
c.Divide(2)
c.cd(1)
ROOT.gPad.SetLeftMargin(0.15)
xframe.GetYaxis().SetTitleOffset(1.6)
xframe.Draw()
c.cd(2)
ROOT.gPad.SetLeftMargin(0.15)
sframe.GetYaxis().SetTitleOffset(1.6)
sframe.Draw()

c.SaveAs("rf111_derivatives.png")
