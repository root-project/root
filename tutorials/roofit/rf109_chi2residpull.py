## \file
## \ingroup tutorial_roofit
## \notebook
## 'BASIC FUNCTIONALITY' RooFit tutorial macro #109
## Calculating chi^2 from histograms and curves in ROOT.RooPlots,
## making histogram of residual and pull distributions
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

# Create observables
x = ROOT.RooRealVar("x", "x", -10, 10)

# Create Gaussian
sigma = ROOT.RooRealVar("sigma", "sigma", 3, 0.1, 10)
mean = ROOT.RooRealVar("mean", "mean", 0, -10, 10)
gauss = ROOT.RooGaussian("gauss", "gauss", x, mean, sigma)

# Generate a sample of 1000 events with sigma=3
data = gauss.generate({x}, 10000)

# Change sigma to 3.15
sigma.setVal(3.15)

# Plot data and slightly distorted model
# ---------------------------------------------------------------------------

# Overlay projection of gauss with sigma=3.15 on data with sigma=3.0
frame1 = x.frame(Title="Data with distorted Gaussian pdf", Bins=40)
data.plotOn(frame1, DataError="SumW2")
gauss.plotOn(frame1)

# Calculate chi^2
# ------------------------------

# Show the chi^2 of the curve w.r.t. the histogram
# If multiple curves or datasets live in the frame you can specify
# the name of the relevant curve and/or dataset in chiSquare()
print("chi^2 = ", frame1.chiSquare())

# Show residual and pull dists
# -------------------------------------------------------

# Construct a histogram with the residuals of the data w.r.t. the curve
hresid = frame1.residHist()

# Construct a histogram with the pulls of the data w.r.t the curve
hpull = frame1.pullHist()

# Create a frame to draw the residual distribution and add the
# distribution to the frame
frame2 = x.frame(Title="Residual Distribution")
frame2.addPlotable(hresid, "P")

# Create a frame to draw the pull distribution and add the distribution to
# the frame
frame3 = x.frame(Title="Pull Distribution")
frame3.addPlotable(hpull, "P")

c = ROOT.TCanvas("rf109_chi2residpull", "rf109_chi2residpull", 900, 300)
c.Divide(3)
c.cd(1)
ROOT.gPad.SetLeftMargin(0.15)
frame1.GetYaxis().SetTitleOffset(1.6)
frame1.Draw()
c.cd(2)
ROOT.gPad.SetLeftMargin(0.15)
frame2.GetYaxis().SetTitleOffset(1.6)
frame2.Draw()
c.cd(3)
ROOT.gPad.SetLeftMargin(0.15)
frame3.GetYaxis().SetTitleOffset(1.6)
frame3.Draw()

c.SaveAs("rf109_chi2residpull.png")
