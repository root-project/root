## \file
## \ingroup tutorial_graphs
## \notebook
##
## Draws three graphs with an exclusion zone.
##
## \macro_image
## \macro_code
##
## \author Olivier Couet
## \translator P. P.


import ROOT
import ctypes

#classes
TMultiGraph = ROOT.TMultiGraph
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


# TCanvas
def exclusiongraph() :

   global c1
   c1 = TCanvas("c1", "Exclusion graphs examples", 200, 10, 600, 400)
   c1.SetGrid()
   
   global mg
   mg = TMultiGraph()
   mg.SetTitle("Exclusion graphs")
   
   n = 35
   xvalues1, xvalues2, xvalues3, yvalues1, yvalues2, yvalues3 = [ to_c([0.]*n)  for _ in range(6) ]

   #   for (Int_t i = 0; i < n; i++) {
   for i in range(0, n, 1):
      xvalues1[i] = i * 0.1
      xvalues2[i] = xvalues1[i]
      xvalues3[i] = xvalues1[i] + .5
      yvalues1[i] = 10 * sin(xvalues1[i])
      yvalues2[i] = 10 * cos(xvalues1[i])
      yvalues3[i] = 10 * sin(xvalues1[i]) - 2
      
   
   global gr1
   gr1 = TGraph(n, xvalues1, yvalues1)
   gr1.SetLineColor(2)
   gr1.SetLineWidth(1504)
   gr1.SetFillStyle(3005)
   

   global gr2
   gr2 = TGraph(n, xvalues2, yvalues2)
   gr2.SetLineColor(4)
   gr2.SetLineWidth(-2002)
   gr2.SetFillStyle(3004)
   gr2.SetFillColor(9)
   

   global gr3
   gr3 = TGraph(n, xvalues3, yvalues3)
   gr3.SetLineColor(5)
   gr3.SetLineWidth(-802)
   gr3.SetFillStyle(3002)
   gr3.SetFillColor(2)
   
   mg.Add(gr1)
   mg.Add(gr2)
   mg.Add(gr3)
   mg.Draw("AC")
   
   return c1
   


if __name__ == "__main__":
   exclusiongraph()
