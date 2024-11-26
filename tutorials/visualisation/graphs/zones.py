## \file
## \ingroup tutorial_graphs
## \notebook
## Example of script showing how to divide a canvas
## into adjacent subpads + axis labels on the top and right side
## of the pads. Original tutorial by Rene Brun.
##
## \macro_image
## \macro_code
##
## \author Alberto Ferro

import ROOT

c1 = ROOT.TCanvas("c1","multipads",900,700)
ROOT.gStyle.SetOptStat(0)

c1.Divide(2,2,0,0)
h1 = ROOT.TH2F("h1","test1",10,0,1,20,0,20)
h2 = ROOT.TH2F("h2","test2",10,0,1,20,0,100)
h3 = ROOT.TH2F("h3","test3",10,0,1,20,-1,1)
h4 = ROOT.TH2F("h4","test4",10,0,1,20,0,1000)

c1.cd(1)
ROOT.gPad.SetTickx(2)
h1.Draw()
c1.cd(2)
ROOT.gPad.SetTickx(2)
ROOT.gPad.SetTicky(2)
h2.GetYaxis().SetLabelOffset(0.01)
h2.Draw()
c1.cd(3)
h3.Draw()
c1.cd(4)
ROOT.gPad.SetTicky(2)
h4.Draw()

