## \file
## \ingroup tutorial_hist
## \notebook
## Fill a 1-D histogram from a parametric function. Original tutorial by Rene Brun.
##
## \macro_image
## \macro_code
##
## \author Alberto Ferro

import ROOT

c1 = ROOT.TCanvas("c1","The FillRandom example",200,10,700,900)
pad1 = ROOT.TPad("pad1","The pad with the function",0.05,0.50,0.95,0.95)
pad2 = ROOT.TPad("pad2","The pad with the histogram",0.05,0.05,0.95,0.45)
pad1.Draw()
pad2.Draw()
pad1.cd()
ROOT.gBenchmark.Start("fillrandom")

form1 = ROOT.TFormula("form1","abs(sin(x)/x)")
sqroot = ROOT.TF1("sqroot","x*gaus(0) + [3]*form1",0,10)
sqroot.SetParameters(10,4,1,20)
pad1.SetGridx()
pad1.SetGridy()
pad1.GetFrame().SetBorderMode(-1)
pad1.GetFrame().SetBorderSize(5)
sqroot.SetLineColor(4)
sqroot.SetLineWidth(6)
sqroot.Draw()
lfunction = ROOT.TPaveLabel(5,39,9.8,46,"The sqroot function")
lfunction.Draw()
c1.Update()

pad2.cd()
pad2.GetFrame().SetBorderMode(-1)
pad2.GetFrame().SetBorderSize(5)
h1f = ROOT.TH1F("h1f","Test random numbers",200,0,10)
h1f.SetFillColor(45)
h1f.FillRandom("sqroot",10000)
h1f.Draw()
c1.Update()

f = ROOT.TFile("fillrandom-py.root","RECREATE")
form1.Write()
sqroot.Write()
h1f.Write()
ROOT.gBenchmark.Show("fillrandom")
