## \file
## \ingroup tutorial_graphs
## \notebook -js
## \preview Create a TGraphSmooth and show the usage of the interpolation function Approx.
##
## See the [TGraphSmooth documentation](https://root.cern/doc/master/classTGraphSmooth.html)
##
## \macro_image
## \macro_code
## \author Christian Stratowa, Jamie Gooding

from ctypes import c_double

import numpy as np
import ROOT

# vC1 = ROOT.TCanvas()
# # grxy = ROOT.TGraph()
# grin = ROOT.TGraph()
# grout = ROOT.TGraph()


def DrawSmooth(pad, title, xt, yt):
    vC1.cd(pad)
    vFrame = ROOT.gPad.DrawFrame(0, 0, 15, 150)
    vFrame.SetTitle(title)
    vFrame.SetTitleSize(0.2)
    vFrame.SetXTitle(xt)
    vFrame.SetYTitle(yt)
    grxy.SetMarkerColor(ROOT.kBlue)
    grxy.SetMarkerStyle(21)
    grxy.SetMarkerSize(0.5)
    grxy.Draw("P")
    grin.SetMarkerColor(ROOT.kRed)
    grin.SetMarkerStyle(5)
    grin.SetMarkerSize(0.7)
    grin.Draw("P")
    grout.DrawClone("LP")


# Test data (square)
n = 11
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.0, 6.0, 8.0, 9.0, 10.0])
y = np.array([1.0, 4.0, 9.0, 16.0, 25.0, 25.0, 36.0, 49.0, 64.0, 81.0, 100.0])

grxy = ROOT.TGraph(n, x, y)

# X values, for which y values should be interpolated
nout = 14
xout = np.array([1.2, 1.7, 2.5, 3.2, 4.4, 5.2, 5.7, 6.5, 7.6, 8.3, 9.7, 10.4, 11.3, 13])

# Create Canvas
vC1 = ROOT.TCanvas("vC1", "square", 200, 10, 700, 700)
vC1.Divide(2, 2)

# Initialize graph with data
grin = ROOT.TGraph(n, x, y)
# Interpolate at equidistant points (use mean for tied x-values)
gs = ROOT.TGraphSmooth("normal")
grout = gs.Approx(grin, "linear")
DrawSmooth(1, "Approx: ties = mean", "X-axis", "Y-axis")

# Re-initialize graph with data
# (since graph points were set to unique vales)
grin = ROOT.TGraph(n, x, y)
# Interpolate at given points xout
grout = gs.Approx(grin, "linear", 14, xout, 0, 130)
DrawSmooth(2, "Approx: ties = mean", "", "")

# Print output variables for given values xout
vNout = grout.GetN()
vXout = c_double()
vYout = c_double()
for k in range(vNout):
    grout.GetPoint(k, vXout, vYout)
    print(f"k= {k}  vXout[k]= {vXout.value}  vYout[k]= {vYout.value}")

# Re-initialize graph with data
grin = ROOT.TGraph(n, x, y)
# Interpolate at equidistant points (use min for tied x-values)
#   _grout = gs.Approx(grin,"linear", 50, 0, 0, 0, 1, 0, "min")_
grout = gs.Approx(grin, "constant", 50, 0, 0, 0, 1, 0.5, "min")
DrawSmooth(3, "Approx: ties = min", "", "")

# Re-initialize graph with data
grin = ROOT.TGraph(n, x, y)
# Interpolate at equidistant points (use max for tied x-values)
grout = gs.Approx(grin, "linear", 14, xout, 0, 0, 2, 0, "max")
DrawSmooth(4, "Approx: ties = max", "", "")
