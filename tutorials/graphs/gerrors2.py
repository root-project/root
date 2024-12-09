## \file
## \ingroup tutorial_graphs
## \notebook -js
##
## This script draws two graphs jointly with their error bars.
##
## \macro_image
## \macro_code
##
## \author Rene Brun
## \translator P. P.


import ROOT
import ctypes

#classes
TCanvas = ROOT.TCanvas
TGraphErrors = ROOT.TGraphErrors
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
def gerrors2() :

   global c1
   c1 = TCanvas("c1", "gerrors2", 200, 10, 700, 500)
   c1.SetGrid()
   
   # draw a frame to define the range
   global hr
   hr = c1.DrawFrame(-0.4, 0, 1.2, 12) # TFrame
   hr.SetXTitle("X title")
   hr.SetYTitle("Y title")
   c1.GetFrame().SetBorderSize(12)
   
   # create first graph
   n1 = 10
   xval1 = [ -0.22, 0.05, 0.25, 0.35, 0.5, 0.61, 0.7, 0.85, 0.89, 0.95 ]
   yval1 = [ 1, 2.9, 5.6, 7.4, 9, 9.6, 8.7, 6.3, 4.5, 1 ]
   ex1 = [ .05, .1, .07, .07, .04, .05, .06, .07, .08, .05 ]
   ey1 = [ .8, .7, .6, .5, .4, .4, .5, .6, .7, .8 ]
   # to c-types
   xval1 = to_c( xval1 )
   yval1 = to_c( yval1 )
   ex1 = to_c( ex1 )
   ey1 = to_c( ey1 )

   global gr1
   gr1 = TGraphErrors(n1, xval1, yval1, ex1, ey1)
   gr1.SetMarkerColor(kBlue)
   gr1.SetMarkerStyle(21)
   gr1.Draw("LP")
   
   # create second graph
   n2 = 10
   xval2 = [ -0.28, 0.005, 0.19, 0.29, 0.45, 0.56, 0.65, 0.80, 0.90, 1.01 ]
   yval2 = [ 0.82, 3.86, 7, 9, 10, 10.55, 9.64, 7.26, 5.42, 2 ]
   ex2 = [ .04, .12, .08, .06, .05, .04, .07, .06, .08, .04 ]
   ey2 = [ .6, .8, .7, .4, .3, .3, .4, .5, .6, .7 ]
   # to c-types
   xval2 = to_c( xval2 )
   yval2 = to_c( yval2 )
   ex2 = to_c( ex2 )
   ey2 = to_c( ey2 )

   global gr2
   gr2 = TGraphErrors(n2, xval2, yval2, ex2, ey2)
   gr2.SetMarkerColor(kRed)
   gr2.SetMarkerStyle(20)
   gr2.Draw("LP")
   


if __name__ == "__main__":
   gerrors2()
