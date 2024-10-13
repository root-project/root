## \file
## \ingroup tutorial_roofit
## \notebook
## Taylor expansion of RooFit functions using the taylorExpand function
##
## \macro_image
## \macro_code
## \macro_output
##
## \date November 2021
## \author Rahul Balasubramanian

import ROOT

# Create functions
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -

x = ROOT.RooRealVar("x", "x", 0.0, -3, 10)

# RooPolyFunc polynomial
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
# x ^ 4 - 5x ^ 3 + 5x ^ 2 + 5x - 6
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
f = ROOT.RooPolyFunc("f", "f", ROOT.RooArgList(x))
f.addTerm(+1, x, 4)
f.addTerm(-5, x, 3)
f.addTerm(+5, x, 2)
f.addTerm(+5, x, 1)
f.addTerm(-6, x, 0)

f = ROOT.RooFormulaVar("f", "f", "pow(@0,4) -5 * pow(@0,3) +5 * pow(@0,2) + 5 * pow(@0,1) - 6", [x])
# taylor expand around x0 = 0
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
x0 = 2.0
taylor_o1 = ROOT.RooPolyFunc.taylorExpand("taylorfunc_o1", "taylor expansion order 1", f, [x], 1, [x0])
ROOT.SetOwnership(taylor_o1, True)
taylor_o2 = ROOT.RooPolyFunc.taylorExpand("taylorfunc_o2", "taylor expansion order 2", f, [x], 2, [x0])
ROOT.SetOwnership(taylor_o2, True)
frame = x.frame(Title="x^{4} - 5x^{3} + 5x^{2} + 5x - 6")
c = ROOT.TCanvas("c", "c", 400, 400)

f.plotOn(frame, Name="f")
taylor_o1.plotOn(frame, Name="taylor_o1", LineColor="kRed", LineStyle="kDashed")
taylor_o2.plotOn(frame, Name="taylor_o2", LineColor="kRed - 9", LineStyle="kDotted")

c.cd()
frame.SetMinimum(-8.0)
frame.SetMaximum(+8.0)
frame.SetYTitle("function value")
frame.Draw()

legend = ROOT.TLegend(0.53, 0.73, 0.86, 0.87)
legend.SetFillColor(ROOT.kWhite)
legend.SetLineColor(ROOT.kWhite)
legend.SetTextSize(0.02)
legend.AddEntry("taylor_o1", "Taylor expansion upto first order", "L")
legend.AddEntry("taylor_o2", "Taylor expansion upto second order", "L")
legend.AddEntry("f", "Polynomial of fourth order", "L")
legend.Draw()

c.Draw()
c.SaveAs("rf710_roopoly.png")
