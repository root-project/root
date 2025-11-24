## \file
## \ingroup tutorial_graphics
## \notebook -js
##
## This scripts draws many arrows.
##
## \macro_image (tcanvas_js)
## \macro_code
##
## \author Rene Brun
## \translator P. P.


import ROOT

#classes
TCanvas = ROOT.TCanvas
TPaveLabel = ROOT.TPaveLabel
TArrow = ROOT.TArrow


# void
def arrows() :

   global c1
   c1 = TCanvas("c1")
   c1.Range(0,0,1,1)
   

   global par
   par = TPaveLabel(0.1,0.8,0.9,0.95,"Examples of various arrows formats")
   par.SetFillColor(42)
   par.Draw()
   

   global ar1
   ar1 = TArrow(0.1,0.1,0.1,0.7)
   ar1.Draw()

   global ar2
   ar2 = TArrow(0.2,0.1,0.2,0.7,0.05,"|>")
   ar2.SetAngle(40)
   ar2.SetLineWidth(2)
   ar2.Draw()

   global ar3
   ar3 = TArrow(0.3,0.1,0.3,0.7,0.05,"<|>")
   ar3.SetAngle(40)
   ar3.SetLineWidth(2)
   ar3.Draw()

   global ar4
   ar4 = TArrow(0.46,0.7,0.82,0.42,0.07,"|>")
   ar4.SetAngle(60)
   ar4.SetLineWidth(2)
   ar4.SetFillColor(2)
   ar4.Draw()

   global ar5
   ar5 = TArrow(0.4,0.25,0.95,0.25,0.15,"<|>")
   ar5.SetAngle(60)
   ar5.SetLineWidth(4)
   ar5.SetLineColor(4)
   ar5.SetFillStyle(3008)
   ar5.SetFillColor(2)
   ar5.Draw()
   


if __name__ == "__main__":
   arrows()
