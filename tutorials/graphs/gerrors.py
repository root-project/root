## \file
## \ingroup tutorial_graphs
## \notebook -js
##
## Draws a graph with error bars.
##
## \macro_image
## \macro_code
##
## \author Rene Brun
## \translator P. P.


import ROOT
import ctypes

#classes
TGraphErrors = ROOT.TGraphErrors
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
def gerrors() :

   global c1
   c1 = TCanvas("c1", "A Simple Graph with error bars", 200, 10, 700, 500)
   
   c1.SetGrid()
   c1.GetFrame().SetBorderSize(12)
   
   n = 10
   x = [ -0.22, 0.05, 0.25, 0.35, 0.5, 0.61, 0.7, 0.85, 0.89, 0.95 ]
   y = [ 1, 2.9, 5.6, 7.4, 9, 9.6, 8.7, 6.3, 4.5, 1 ]
   ex = [ .05, .1, .07, .07, .04, .05, .06, .07, .08, .05 ]
   ey = [ .8, .7, .6, .5, .4, .4, .5, .6, .7, .8 ]
   # to c-types
   x = to_c( x )
   y = to_c( y )
   ex = to_c( ex )
   ey = to_c( ey )

   global gr
   gr = TGraphErrors(n, x, y, ex, ey)

   gr.SetTitle("TGraphErrors Example")
   gr.SetMarkerColor(4)
   gr.SetMarkerStyle(21)

   gr.Draw("ALP")
   
   c1.Update()
   


if __name__ == "__main__":
   gerrors()
