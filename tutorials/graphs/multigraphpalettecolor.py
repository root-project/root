## \file
## \ingroup tutorial_graphs
## \notebook
##
## Palette coloring feature for multi-graphs is activated thanks to the options `PFC`
## (Palette Fill Color), `PLC` (Palette Line Color) and `AMC` (Palette Marker Color).
## When one of these options is given to `TMultiGraph.Draw` the `TGraph`s objects in the
## `TMultiGraph`-class get their color from the current color palette defined by
## `gStyle.SetPalette(...)`. The color is determined according to the number of
## `TGraph`s objects.
##
## In this example, four graphs are displayed with palette coloring feature for lines and
## and markers. The color of each graph is picked inside the default palette `kBird`.
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
TMultiGraph = ROOT.TMultiGraph
TMath = ROOT.TMath
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
def multigraphpalettecolor() :

   global mg
   mg = TMultiGraph()
   

   global gr1, gr2, gr3, gr4
   gr1 = TGraph()
   gr1.SetMarkerStyle(20)

   gr2 = TGraph()
   gr2.SetMarkerStyle(21)

   gr3 = TGraph()
   gr3.SetMarkerStyle(23)

   gr4 = TGraph()
   gr4.SetMarkerStyle(24)
   
   dx = 6.28 / 100
   x = -3.14
   
   #   for (int i = 0; i <= 100; i++) {
   for i in range(0, 100 + 1, 1):
      x = x + dx
      gr1.SetPoint(i, x, 2. * TMath.Sin(x))
      gr2.SetPoint(i, x, TMath.Cos(x))
      gr3.SetPoint(i, x, TMath.Cos(x * x))
      gr4.SetPoint(i, x, TMath.Cos(x * x * x))
      
   
   mg.Add(gr4, "PL")
   mg.Add(gr3, "PL")
   mg.Add(gr2, "*L")
   mg.Add(gr1, "PL")
   
   mg.Draw("A pmc plc")
   


if __name__ == "__main__":
   multigraphpalettecolor()
