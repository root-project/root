## \file
## \ingroup tutorial_graphs
## \notebook -js
##
## Draws a graph with multiple $y$ axis errors.
##
## \macro_image
## \macro_code
##
## \author Simon Spies
## \translator P. P.


import ROOT
import ctypes

#classes
TGraphMultiErrors = ROOT.TGraphMultiErrors
TCanvas = ROOT.TCanvas
TH1F = ROOT.TH1F
TGraph = ROOT.TGraph
TLatex = ROOT.TLatex

TCanvas = ROOT.TCanvas
TPaveLabel = ROOT.TPaveLabel
TPavesText = ROOT.TPavesText
TPaveText = ROOT.TPaveText
TText = ROOT.TText
TArrow = ROOT.TArrow
TWbox = ROOT.TWbox
TPad = ROOT.TPad
TBox = ROOT.TBox
TPad = ROOT.TPad
#maths
sin = ROOT.sin
cos = ROOT.cos
sqrt = ROOT.sqrt

#types
Double_t = ROOT.Double_t
Float_t = ROOT.Float_t
Int_t = ROOT.Int_t
nullptr = ROOT.nullptr
c_double = ctypes.c_double

#utils
def to_c( ls ):
   return (c_double * len(ls) )( * ls )
def printf(string, *args):
   print( string % args )
def sprintf(buffer, string, *args):
   buffer = string % args 
   return buffer

#constants
kBlue = ROOT.kBlue
kRed = ROOT.kRed
kGreen = ROOT.kGreen

#globals
gStyle = ROOT.gStyle
gPad = ROOT.gPad
gRandom = ROOT.gRandom
gBenchmark = ROOT.gBenchmark
gROOT = ROOT.gROOT



# void
def gmultierrors() :

   global c1
   c1 = TCanvas("c1", "A Simple Graph with multiple y-errors", 200, 10, 700, 500)
   c1.SetGrid()
   c1.GetFrame().SetBorderSize(12)
   
   np = 5
   x = [ 0, 1, 2, 3, 4 ]
   y = [ 0, 2, 4, 1, 3 ]
   exl = [ 0.3, 0.3, 0.3, 0.3, 0.3 ]
   exh = [ 0.3, 0.3, 0.3, 0.3, 0.3 ]
   eylstat = [ 1, 0.5, 1, 0.5, 1 ]
   eyhstat = [ 0.5, 1, 0.5, 1, 0.5 ]
   eylsys = [ 0.5, 0.4, 0.8, 0.3, 1.2 ]
   eyhsys = [ 0.6, 0.7, 0.6, 0.4, 0.8 ]
   # c-types
   x = to_c( x )
   y = to_c( y )
   exl = to_c( exl )
   exh = to_c( exh )
   eylstat = to_c( eylstat )
   eyhstat = to_c( eyhstat )
   eylsys = to_c( eylsys )
   eyhsys = to_c( eyhsys )
   

   global gme
   gme = TGraphMultiErrors("gme", "TGraphMultiErrors Example", np, x, y, exl, exh, eylstat, eyhstat)

   # Adding $y$ error axis to.
   gme.AddYError(np, eylsys, eyhsys)

   # Setting-up.
   gme.SetMarkerStyle(20)
   gme.SetLineColor(kRed)
   gme.GetAttLine(0).SetLineColor(kRed)
   gme.GetAttLine(1).SetLineColor(kBlue)
   gme.GetAttFill(1).SetFillStyle(0)
   
   # Draw graph and x erros are drawn with "APS"
   # Stat Errors drawn with "Z"
   # Sys Errors drawn with "5 s=0.5"
   gme.Draw("APS ; Z ; 5 s=0.5")
   
   c1.Update()
   


if __name__ == "__main__":
   gmultierrors()
