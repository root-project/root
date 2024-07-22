## \file
## \ingroup tutorial_graphs
## \notebook
##
## This script creates and draws a TMultiGraph.
##
## \macro_image
## \macro_code
##
## \author Rene Brun
## \translator P. P.


import ROOT
import ctypes

#classes
TMultiGraph = ROOT.TMultiGraph
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
def multigraph() :

   gStyle.SetOptFit()
   #gStyle.SetOptFit(False)

   global c1
   c1 = TCanvas("c1", "multigraph", 700, 500)
   c1.SetGrid()
   
   # draw a frame to define the range

   global mg
   mg = TMultiGraph()
   
   # create first graph
   n1 = 10
   px1 = [ -0.1, 0.05, 0.25, 0.35, 0.5, 0.61, 0.7, 0.85, 0.89, 0.95 ]
   py1 = [ -1, 2.9, 5.6, 7.4, 9, 9.6, 8.7, 6.3, 4.5, 1 ]
   ex1 = [ .05, .1, .07, .07, .04, .05, .06, .07, .08, .05 ]
   ey1 = [ .8, .7, .6, .5, .4, .4, .5, .6, .7, .8 ]
   #to c-types
   px1 = to_c( px1 )
   py1 = to_c( py1 )
   ex1 = to_c( ex1 )
   ey1 = to_c( ey1 )


   global gr1
   gr1 = TGraphErrors(n1, px1, py1, ex1, ey1)
   gr1.SetMarkerColor(kBlue)
   gr1.SetMarkerStyle(21)
   gr1.Fit("pol6", "q")
   #gr1.Fit("pol6", "S")
   #gr1.Fit("pol6", "Q")
   #gr1.Fit("pol6", "L")
   #gr1.Fit("pol6", "M")
   #gr1.Fit("pol6", "V")
   # OPTIONS:
   # "S": Perform a robust fit.
   # "Q": Quiet mode, suppresses output.
   # "L": Use log-likelihood method for chi-square minimization.
   # "V": Verbose mode, prints more information during the fit.
   # "M": More, improve fit results.


   mg.Add(gr1)
   
   # create second graph
   n2 = 10
   x2 = [ -0.28, 0.005, 0.19, 0.29, 0.45, 0.56, 0.65, 0.80, 0.90, 1.01 ]
   y2 = [ 2.1, 3.86, 7, 9, 10, 10.55, 9.64, 7.26, 5.42, 2 ]
   ex2 = [ .04, .12, .08, .06, .05, .04, .07, .06, .08, .04 ]
   ey2 = [ .6, .8, .7, .4, .3, .3, .4, .5, .6, .7 ]
   #to c-types
   x2 = to_c( x2 )
   y2 = to_c( y2 )
   ex2 = to_c( ex2 )
   ey2 = to_c( ey2 )


   global gr2
   gr2 = TGraphErrors(n2, x2, y2, ex2, ey2)
   gr2.SetMarkerColor(kRed)
   gr2.SetMarkerStyle(20)
   gr2.Fit("pol5", "q")
   #gr2.Fit("pol5", "S")
   #gr2.Fit("pol5", "Q")
   #gr2.Fit("pol5", "L")
   #gr2.Fit("pol5", "M")
   #gr2.Fit("pol5", "V")
   # OPTIONS:
   # "S": Perform a robust fit.
   # "Q": Quiet mode, suppresses output.
   # "L": Use log-likelihood method for chi-square minimization.
   # "V": Verbose mode, prints more information during the fit.
   # "M": More, improve fit results.
   
   mg.Add(gr2)
   
   mg.Draw("ap")
   
   # force drawing of canvas to generate the fit TPaveStats
   c1.Update()
   
   global stats1, stats2
   stats1 = gr1.GetListOfFunctions().FindObject("stats") # TPaveStats
   stats2 = gr2.GetListOfFunctions().FindObject("stats") # TPaveStats
   
   if stats1 and stats2:

      stats1.SetTextColor(kBlue)
      stats2.SetTextColor(kRed)
      stats1.SetX1NDC(0.12)
      stats1.SetX2NDC(0.32)
      stats1.SetY1NDC(0.75)
      stats2.SetX1NDC(0.72)
      stats2.SetX2NDC(0.92)
      stats2.SetY1NDC(0.78)

      c1.Modified()
   #Note:
   #     The statistic results differ from the C-version.
   #     Could be something in py-float to c_double conversion.
   #     
   


if __name__ == "__main__":
   multigraph()
