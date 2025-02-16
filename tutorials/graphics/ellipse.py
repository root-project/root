## \file
## \ingroup tutorial_graphics
## \notebook -js
## This script draws many ellipses.
##
## \macro_image
## \macro_code
##
## \author Rene Brun
## \translator P. P.


import ROOT

#classes
TCanvas = ROOT.TCanvas
TPaveLabel = ROOT.TPaveLabel
TEllipse = ROOT.TEllipse


# TCanvas
def ellipse():

   global c1
   c1 = TCanvas("c1")
   c1.Range(0,0,1,1)

   global pel
   pel = TPaveLabel(0.1,0.8,0.9,0.95,"Examples of Ellipses")
   pel.SetFillColor(42)
   pel.Draw()

   global el1
   el1 = TEllipse(0.25,0.25,.1,.2)
   el1.Draw()

   global el2
   el2 = TEllipse(0.25,0.6,.2,.1)
   el2.SetFillColor(6)
   el2.SetFillStyle(3008)
   el2.Draw()

   global el3
   el3 = TEllipse(0.75,0.6,.2,.1,45,315)
   el3.SetFillColor(2)
   el3.SetFillStyle(1001)
   el3.SetLineColor(4)
   el3.Draw()

   global el4
   el4 = TEllipse(0.75,0.25,.2,.15,45,315,62)
   el4.SetFillColor(5)
   el4.SetFillStyle(1001)
   el4.SetLineColor(4)
   el4.SetLineWidth(6)
   el4.Draw()

   return c1
   


if __name__ == "__main__":
   ellipse()
