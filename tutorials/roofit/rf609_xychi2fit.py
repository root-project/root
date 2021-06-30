## \file
## \ingroup tutorial_roofit
## \notebook
## Likelihood and minimization: setting up a chi^2 fit to an unbinned dataset with X,Y,err(Y)
## values (and optionally err(X) values)
##
## \macro_code
##
## \date February 2018
## \authors Clemens Lange, Wouter Verkerke (C++ version)

import ROOT
import math


# Create dataset with X and Y values
# -------------------------------------------------------------------

# Make weighted XY dataset with asymmetric errors stored
# The StoreError() argument is essential as it makes
# the dataset store the error in addition to the values
# of the observables. If errors on one or more observables
# are asymmetric, can store the asymmetric error
# using the StoreAsymError() argument

x = ROOT.RooRealVar("x", "x", -11, 11)
y = ROOT.RooRealVar("y", "y", -10, 200)
dxy = ROOT.RooDataSet("dxy", "dxy", ROOT.RooArgSet(x, y), StoreError=(ROOT.RooArgSet(x, y)))

# Fill an example dataset with X,err(X),Y,err(Y) values
for i in range(10):
    x.setVal(-10 + 2 * i)
    x.setError((0.5 / 1.0) if (i < 5) else (1.0 / 1.0))

    # Set Y value and error
    y.setVal(x.getVal() * x.getVal() + 4 * abs(ROOT.gRandom.Gaus()))
    y.setError(math.sqrt(y.getVal()))

    dxy.add(ROOT.RooArgSet(x, y))

# Perform chi2 fit to X +/- dX and Y +/- dY values
# ---------------------------------------------------------------------------------------

# Make fit function
a = ROOT.RooRealVar("a", "a", 0.0, -10, 10)
b = ROOT.RooRealVar("b", "b", 0.0, -100, 100)
f = ROOT.RooPolyVar("f", "f", x, ROOT.RooArgList(b, a, ROOT.RooFit.RooConst(1)))

# Plot dataset in X-Y interpretation
frame = x.frame(Title="Chi^2 fit of function set of (X#pmdX,Y#pmdY) values")
dxy.plotOnXY(frame, YVar=y)

# Fit chi^2 using X and Y errors
f.chi2FitTo(dxy, YVar=y)

# Overlay fitted function
f.plotOn(frame)

# Alternative: fit chi^2 integrating f(x) over ranges defined by X errors, rather
# than taking point at center of bin
f.chi2FitTo(dxy, YVar=y, Integrate=True)

# Overlay alternate fit result
f.plotOn(frame, LineStyle="--", LineColor="r")

# Draw the plot on a canvas
c = ROOT.TCanvas("rf609_xychi2fit", "rf609_xychi2fit", 600, 600)
ROOT.gPad.SetLeftMargin(0.15)
frame.GetYaxis().SetTitleOffset(1.4)
frame.Draw()

c.SaveAs("rf609_xychi2fit.png")
