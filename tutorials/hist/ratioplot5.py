## \file
## \ingroup tutorial_hist
## \notebook
## Example that shows how you can set the colors of the confidence interval bands by using
## the method `TRatioPlot::SetConfidenceIntervalColors`.
## Inspired by the tutorial of Paul Gessinger.
##
## \macro_image
## \macro_code
##
## \author Alberto Ferro

import ROOT

ROOT.gStyle.SetOptStat(0)

c1 = ROOT.TCanvas("c1", "fit residual simple")
h1 = ROOT.TH1D("h1", "h1", 50, -5, 5)
h1.FillRandom("gaus", 2000)
h1.Fit("gaus")
h1.GetXaxis().SetTitle("x")
h1.GetYaxis().SetTitle("y")

rp1 = ROOT.TRatioPlot(h1)
rp1.SetConfidenceIntervalColors(ROOT.kBlue, ROOT.kRed)
rp1.Draw()
c1.Update()
