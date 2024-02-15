# \file
# \ingroup tutorial_roostats
# \notebook -js
#
# \macro_image
# \macro_output
# \macro_code
#
# \author Lorenzo Moneta

from ROOT import RooWorkspace, kRed, kGreen, kBlue, kYellow, kDashed, kDotted
from ROOT.RooFit import MarkerColor, LineColor, LineStyle

w = RooWorkspace("w")
# k <2, must use sum
w.factory("NonCentralChiSquare::nc(x[0,50],k[1.99,0,5],lambda[5])")
# kk > 2 can use bessel
w.factory("NonCentralChiSquare::ncc(x,kk[2.01,0,5],lambda)")
# kk > 2, force sum
w.factory("NonCentralChiSquare::nccc(x,kk,lambda)")
w.pdf("nccc").SetForceSum(True)

# a normal "central" chi-square for comparison when lambda->0
w.factory("ChiSquarePdf::cs(x,k)")

# w.var("kk").setVal(4.) # test a large kk

ncdata = w.pdf("nc").generate(w.var("x"), 100)
csdata = w.pdf("cs").generate(w.var("x"), 100)
plot = w.var("x").frame()
ncdata.plotOn(plot, MarkerColor(kRed))
csdata.plotOn(plot, MarkerColor(kBlue))
w.pdf("nc").plotOn(plot, LineColor(kRed))
w.pdf("ncc").plotOn(plot, LineColor(kGreen))
w.pdf("nccc").plotOn(plot, LineColor(kYellow), LineStyle(kDashed))
w.pdf("cs").plotOn(plot, LineColor(kBlue), LineStyle(kDotted))
plot.Draw()
