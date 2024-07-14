## \file
## \ingroup tutorial_graphics
## \notebook
## This script draws many crowns.
##
## \macro_image
## \macro_code
##
## \author Olivier Couet
## \translator P. P.


import ROOT

TCanvas = ROOT.TCanvas
TCrown = ROOT.TCrown



# TCanvas
def crown() :

   global c1, cr1

   global c1
   c1 = TCanvas("c1","c1",400,400)

   global cr1
   cr1 = TCrown(.5,.5,.3,.4)
   cr1.SetLineStyle(2)
   cr1.SetLineWidth(4)
   cr1.Draw()

   global cr2
   cr2 = TCrown(.5,.5,.2,.3,45,315)
   cr2.SetFillColor(38)
   cr2.SetFillStyle(3010)
   cr2.Draw()

   global cr3
   cr3 = TCrown(.5,.5,.2,.3,-45,45)
   cr3.SetFillColor(50)
   cr3.SetFillStyle(3025)
   cr3.Draw()

   global cr4
   cr4 = TCrown(.5,.5,.0,.2)
   cr4.SetFillColor(4)
   cr4.SetFillStyle(3008)
   cr4.Draw()
   return c1
   


if __name__ == "__main__":
   crown()
