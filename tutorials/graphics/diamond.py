## \file
## \ingroup tutorial_graphics
## \notebook
##
## Drawing a diamond.
##
## \macro_image
## \macro_code
##
## \author Olivier Couet
## \translator P. P.


import ROOT

TCanvas = ROOT.TCanvas
TDiamond = ROOT.TDiamond


# TCanvas
def diamond() :

   global c, d
   c = TCanvas("c")
   d = TDiamond(.05,.1,.95,.8)
   
   d.AddText("A TDiamond-class can contain any text.")
   
   d.Draw()
   return c
   


if __name__ == "__main__":
   diamond()
