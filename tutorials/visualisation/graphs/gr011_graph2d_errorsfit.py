## \file
## \ingroup tutorial_graphs
## \notebook
## \preview Create, draw and fit a TGraph2DErrors. See the [TGraph2DErrors documentation](https://root.cern/doc/master/classTGraph2DErrors.html)
##
## \macro_image
## \macro_code
## \author Olivier Couet, Jamie Gooding

from ctypes import c_double

import ROOT

c1 = ROOT.TCanvas("c1")

e = 0.3
nd = 500

# To generate some random data to put into the graph
r = ROOT.TRandom()
f2 = ROOT.TF2("f2", "1000*(([0]*sin(x)/x)*([1]*sin(y)/y))+200", -6, 6, -6, 6)
f2.SetParameters(1, 1)

dte = ROOT.TGraph2DErrors(nd)

# Fill the 2D graph. It was created only specifying the number of points, so all
# elements are empty. We now "fill" the values and errors with SetPoint and SetPointError.
# Note that the first point has index zero
x = c_double()
y = c_double()
zmax = 0
for i in range(nd):
    f2.GetRandom2(x, y)
    rnd = r.Uniform(-e, e)  # Generate a random number in [-e,e]
    z = f2.Eval(x, y) * (1 + rnd)
    if z > zmax:
        zmax = z
    dte.SetPoint(i, x, y, z)
    ex = 0.05 * r.Rndm()
    ey = 0.05 * r.Rndm()
    ez = abs(z * rnd)
    dte.SetPointError(i, ex, ey, ez)

# If the fit is not needed, just draw dte here and skip the lines below
# dte.Draw("A p0")

# To do the fit we use a function, in this example the same f2 from above
f2.SetParameters(0.5, 1.5)
dte.Fit(f2)
fit2 = dte.FindObject("f2")
fit2.SetTitle("Minuit fit result on the Graph2DErrors points")
fit2.SetMaximum(zmax)
ROOT.gStyle.SetHistTopMargin(0)
fit2.SetLineColor(1)
fit2.SetLineWidth(1)
fit2.Draw("surf1")
dte.Draw("same p0")
