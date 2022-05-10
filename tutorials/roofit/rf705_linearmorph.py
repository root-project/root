## \file
## \ingroup tutorial_roofit
## \notebook
##
## 'SPECIAL PDFS' RooFit tutorial macro #705
##
## Linear interpolation between p.d.f shapes using the 'Alex Read' algorithm
##
## \macro_code
##
## \date February 2018
## \authors Clemens Lange, Wouter Verkerke (C version)


import ROOT


# Create end point pdf shapes
# ------------------------------------------------------

# Observable
x = ROOT.RooRealVar("x", "x", -20, 20)

# Lower end point shape: a Gaussian
g1mean = ROOT.RooRealVar("g1mean", "g1mean", -10)
g1 = ROOT.RooGaussian("g1", "g1", x, g1mean, ROOT.RooFit.RooConst(2))

# Upper end point shape: a Polynomial
g2 = ROOT.RooPolynomial("g2", "g2", x, [-0.03, -0.001])

# Create interpolating pdf
# -----------------------------------------------

# Create interpolation variable
alpha = ROOT.RooRealVar("alpha", "alpha", 0, 1.0)

# Specify sampling density on observable and interpolation variable
x.setBins(1000, "cache")
alpha.setBins(50, "cache")

# Construct interpolating pdf in (x,a) represent g1(x) at a=a_min
# and g2(x) at a=a_max
lmorph = ROOT.RooIntegralMorph("lmorph", "lmorph", g1, g2, x, alpha)

# Plot interpolating pdf aat various alphas   a l p h a
# -----------------------------------------------------------------------------

# Show end points as blue curves
frame1 = x.frame()
g1.plotOn(frame1)
g2.plotOn(frame1)

# Show interpolated shapes in red
alpha.setVal(0.125)
lmorph.plotOn(frame1, LineColor="r")
alpha.setVal(0.25)
lmorph.plotOn(frame1, LineColor="r")
alpha.setVal(0.375)
lmorph.plotOn(frame1, LineColor="r")
alpha.setVal(0.50)
lmorph.plotOn(frame1, LineColor="r")
alpha.setVal(0.625)
lmorph.plotOn(frame1, LineColor="r")
alpha.setVal(0.75)
lmorph.plotOn(frame1, LineColor="r")
alpha.setVal(0.875)
lmorph.plotOn(frame1, LineColor="r")
alpha.setVal(0.95)
lmorph.plotOn(frame1, LineColor="r")

# Show 2D distribution of pdf(x,alpha)
# -----------------------------------------------------------------------

# Create 2D histogram
hh = lmorph.createHistogram("hh", x, Binning=40, YVar=dict(var=alpha, Binning=40))
hh.SetLineColor(ROOT.kBlue)

# Fit pdf to dataset with alpha=0.8
# -----------------------------------------------------------------

# Generate a toy dataset alpha = 0.8
alpha.setVal(0.8)
data = lmorph.generate({x}, 1000)

# Fit pdf to toy data
lmorph.setCacheAlpha(True)
lmorph.fitTo(data, Verbose=True)

# Plot fitted pdf and data overlaid
frame2 = x.frame(Bins=100)
data.plotOn(frame2)
lmorph.plotOn(frame2)

# Scan -log(L) vs alpha
# -----------------------------------------

# Show scan -log(L) of dataset w.r.t alpha
frame3 = alpha.frame(Bins=100, Range=(0.1, 0.9))

# Make 2D pdf of histogram
nll = lmorph.createNLL(data)
nll.plotOn(frame3, ShiftToZero=True)

lmorph.setCacheAlpha(False)

c = ROOT.TCanvas("rf705_linearmorph", "rf705_linearmorph", 800, 800)
c.Divide(2, 2)
c.cd(1)
ROOT.gPad.SetLeftMargin(0.15)
frame1.GetYaxis().SetTitleOffset(1.6)
frame1.Draw()
c.cd(2)
ROOT.gPad.SetLeftMargin(0.20)
hh.GetZaxis().SetTitleOffset(2.5)
hh.Draw("surf")
c.cd(3)
ROOT.gPad.SetLeftMargin(0.15)
frame3.GetYaxis().SetTitleOffset(1.4)
frame3.Draw()
c.cd(4)
ROOT.gPad.SetLeftMargin(0.15)
frame2.GetYaxis().SetTitleOffset(1.4)
frame2.Draw()

c.SaveAs("rf705_linearmorph.png")
