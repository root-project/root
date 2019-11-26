## \file
## \ingroup tutorial_math
## \notebook
## Example of first few Legendre Polynomials. Inspired by work of Lorenzo Moneta.
##
## \macro_image
## \macro_code
##
## \author Alberto Ferro, Massimiliano Galli


import ROOT

ROOT.gSystem.Load("libMathMore")
Canvas = ROOT.TCanvas("DistCanvas", "Legendre polynomials example", 10, 10, 750, 600)
Canvas.SetGrid()
leg = ROOT.TLegend(0.5, 0.7, 0.4, 0.89)

L = []
for nu in range(5):
    f = ROOT.TF1("L_0", "ROOT::Math::legendre([0],x)", -1, 1)
    f.SetParameters(nu, 0.0)
    f.SetLineStyle(1)
    f.SetLineWidth(2)
    f.SetLineColor(nu+1)
    L.append(f)

L[0].SetMaximum(1)
L[0].SetMinimum(-1)
L[0].SetTitle("Legendre polynomials")

for idx, val in enumerate(L):
    leg.AddEntry(val, " L_{}(x)".format(idx), "l")
    if idx == 0:
        val.Draw()
    else:
        val.Draw("same")

leg.Draw("same")


