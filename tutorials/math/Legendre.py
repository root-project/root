## \file
## \ingroup tutorial_math
## \notebook
## Example of first few Legendre Polynomials. Inspired by work of Lorenzo Moneta.
##
## \macro_image
## \macro_code
##
## \author Alberto Ferro


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
leg.AddEntry(L[0].Draw(), " L_{0}(x)", "l")
leg.AddEntry(L[1].Draw("same"), " L_{1}(x)", "l")
leg.AddEntry(L[2].Draw("same"), " L_{2}(x)", "l")
leg.AddEntry(L[3].Draw("same"), " L_{3}(x)", "l")
leg.AddEntry(L[4].Draw("same"), " L_{4}(x)", "l")
leg.Draw()


