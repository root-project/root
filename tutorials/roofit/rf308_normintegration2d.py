## \file
## \ingroup tutorial_roofit
## \notebook
## Multidimensional models: normalization and  integration of pdfs, construction of
## cumulative distribution functions from pdfs in two dimensions
##
## \macro_code
##
## \date February 2018
## \authors Clemens Lange, Wouter Verkerke (C++ version)

from __future__ import print_function
import ROOT

# Set up model
# ---------------------

# Create observables x,y
x = ROOT.RooRealVar("x", "x", -10, 10)
y = ROOT.RooRealVar("y", "y", -10, 10)

# Create pdf gaussx(x,-2,3), gaussy(y,2,2)
gx = ROOT.RooGaussian("gx", "gx", x, ROOT.RooFit.RooConst(-2), ROOT.RooFit.RooConst(3))
gy = ROOT.RooGaussian("gy", "gy", y, ROOT.RooFit.RooConst(+2), ROOT.RooFit.RooConst(2))

# gxy = gx(x)*gy(y)
gxy = ROOT.RooProdPdf("gxy", "gxy", [gx, gy])

# Retrieve raw & normalized values of RooFit pdfs
# --------------------------------------------------------------------------------------------------

# Return 'raw' unnormalized value of gx
print("gxy = ", gxy.getVal())

# Return value of gxy normalized over x _and_ y in range [-10,10]
nset_xy = {x, y}
print("gx_Norm[x,y] = ", gxy.getVal(nset_xy))

# Create object representing integral over gx
# which is used to calculate  gx_Norm[x,y] == gx / gx_Int[x,y]
x_and_y = {x, y}
igxy = gxy.createIntegral(x_and_y)
print("gx_Int[x,y] = ", igxy.getVal())

# NB: it is also possible to do the following

# Return value of gxy normalized over x in range [-10,10] (i.e. treating y
# as parameter)
nset_x = {x}
print("gx_Norm[x] = ", gxy.getVal(nset_x))

# Return value of gxy normalized over y in range [-10,10] (i.e. treating x
# as parameter)
nset_y = {y}
print("gx_Norm[y] = ", gxy.getVal(nset_y))

# Integarte normalizes pdf over subrange
# ----------------------------------------------------------------------------

# Define a range named "signal" in x from -5,5
x.setRange("signal", -5, 5)
y.setRange("signal", -3, 3)

# Create an integral of gxy_Norm[x,y] over x and y in range "signal"
# ROOT.This is the fraction of of pdf gxy_Norm[x,y] which is in the
# range named "signal"

igxy_sig = gxy.createIntegral(x_and_y, NormSet=x_and_y, Range="signal")
print("gx_Int[x,y|signal]_Norm[x,y] = ", igxy_sig.getVal())

# Construct cumulative distribution function from pdf
# -----------------------------------------------------------------------------------------------------

# Create the cumulative distribution function of gx
# i.e. calculate Int[-10,x] gx(x') dx'
gxy_cdf = gxy.createCdf({x, y})

# Plot cdf of gx versus x
hh_cdf = gxy_cdf.createHistogram("hh_cdf", x, Binning=40, YVar=dict(var=y, Binning=40))
hh_cdf.SetLineColor(ROOT.kBlue)

c = ROOT.TCanvas("rf308_normintegration2d", "rf308_normintegration2d", 600, 600)
ROOT.gPad.SetLeftMargin(0.15)
hh_cdf.GetZaxis().SetTitleOffset(1.8)
hh_cdf.Draw("surf")

c.SaveAs("rf308_normintegration2d.png")
