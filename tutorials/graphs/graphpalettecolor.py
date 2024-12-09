## \file
## \ingroup tutorial_graphs
## \notebook
##
## The palette coloring for graphs is activated thanks to the options: `PFC` (Palette Fill
## Color), `PLC` (Palette Line Color) and `AMC` (Palette Marker Color). When
## one of these options is given to `TGraph.Draw`, the `TGraph` get its color
## from the current-color-palette defined by `gStyle->SetPalette(...)`. This color
## is determined by the number of objects has the palette-coloring in
## the current pad.
##
## In this example, five graphs are displayed with palette-coloring for lines and
## a filled area. The graphs are drawn with curves (`C` option); and one can see
## which color of each graph is picked inside the palette `kSolar`. The
## same is visible in filled polygons in the automatically-built-legend.
##
## \macro_image
## \macro_code
##
## \author Olivier Couet
## \translator P. P.


import ROOT
import ctypes

#classes
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
#
kSolar = ROOT.kSolar
#
kCircle = ROOT.kCircle
kOpenSquare = ROOT.kOpenSquare
kFullSquare = ROOT.kFullSquare

#globals
gStyle = ROOT.gStyle
gPad = ROOT.gPad
gRandom = ROOT.gRandom
gBenchmark = ROOT.gBenchmark
gROOT = ROOT.gROOT



# void
def graphpalettecolor() :
   
   gStyle.SetOptTitle(False)
   gStyle.SetPalette(kSolar)
   
   x  = [ 1, 2, 3, 4, 5 ]
   y1 = [ 1.0, 2.0, 1.0, 2.5, 3.0 ]
   y2 = [ 1.1, 2.1, 1.1, 2.6, 3.1 ]
   y3 = [ 1.2, 2.2, 1.2, 2.7, 3.2 ]
   y4 = [ 1.3, 2.3, 1.3, 2.8, 3.3 ]
   y5 = [ 1.4, 2.4, 1.4, 2.9, 3.4 ]
   #to C-types
   x = to_c( x )
   y1 = to_c( y1 )
   y2 = to_c( y2 )
   y3 = to_c( y3 )
   y4 = to_c( y4 )
   y5 = to_c( y5 )
   
   global g1, g2, g3, g4, g5 
   g1 = TGraph(5, x, y1)
   g1.SetTitle("Graph with a red star")
   g2 = TGraph(5, x, y2)
   g2.SetTitle("Graph with a circular marker")
   g3 = TGraph(5, x, y3)
   g3.SetTitle("Graph with an open square marker")
   g4 = TGraph(5, x, y4)
   g4.SetTitle("Graph with a blue star")
   g5 = TGraph(5, x, y5)
   g5.SetTitle("Graph with a full square marker")
   
   g1.SetLineWidth(3)
   g1.SetMarkerColor(kRed)
   g2.SetLineWidth(3)
   g2.SetMarkerStyle(kCircle)
   g3.SetLineWidth(3)
   g3.SetMarkerStyle(kOpenSquare)
   g4.SetLineWidth(3)
   g4.SetMarkerColor(kBlue)
   g5.SetLineWidth(3)
   g5.SetMarkerStyle(kFullSquare)
   
   g1.Draw("CA* PLC PFC")
   g2.Draw("PC  PLC PFC")
   g3.Draw("PC  PLC PFC")
   g4.Draw("*C  PLC PFC")
   g5.Draw("PC  PLC PFC")
   
   gPad.BuildLegend()
   


if __name__ == "__main__":
   graphpalettecolor()
