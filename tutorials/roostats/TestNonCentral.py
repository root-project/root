# \file
# \ingroup tutorial_roostats
# \notebook -js
#
# \macro_image
# \macro_output
# \macro_code
#
# \author Lorenzo Moneta

import ROOT

ws = ROOT.RooWorkspace("w")
# k <2, must use sum
ws.factory("NonCentralChiSquare::nc(x[0,50],k[1.99,0,5],lambda[5])")
# kk > 2 can use bessel
ws.factory("NonCentralChiSquare::ncc(x,kk[2.01,0,5],lambda)")
# kk > 2, force sum
ws.factory("NonCentralChiSquare::nccc(x,kk,lambda)")
ws["nccc"].SetForceSum(True)

# a normal "central" chi-square for comparison when lambda->0
ws.factory("ChiSquarePdf::cs(x,k)")

# w.var("kk").setVal(4.) # test a large kk

ncdata = ws["nc"].generate(ws["x"], 100)
csdata = ws["cs"].generate(ws["x"], 100)
plot = ws["x"].frame()
ncdata.plotOn(plot, MarkerColor="r")
csdata.plotOn(plot, MarkerColor="b")
ws["nc"].plotOn(plot, LineColor="r")
ws["ncc"].plotOn(plot, LineColor="g")
ws["nccc"].plotOn(plot, LineColor="y", LineStyle="--")
ws["cs"].plotOn(plot, LineColor="b", LineStyle=":")
plot.Draw()
