## \file
## \ingroup tutorial_graphs
## \notebook
##
## Draws a scatter plot.
##
## \macro_image
## \macro_code
##
## \author Olivier Couet
## \translator P. P.


import ROOT
import ctypes

#classes
TRandom = ROOT.TRandom
TScatter = ROOT.TScatter

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
kBird = ROOT.kBird
 
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
def scatter() :

   gStyle.SetPalette(kBird, 0, 0.6); # define a transparent palette

   global canvas
   canvas = TCanvas()
   
   n = 100
   x = to_c( [ Double_t() for _ in range(n) ] )
   y = to_c( [ Double_t() for _ in range(n) ] )
   c = to_c( [ Double_t() for _ in range(n) ] )
   s = to_c( [ Double_t() for _ in range(n) ] )
   
   # Define four random data set
   global r
   r = TRandom()
   #   for (int i = 0; i < n; i++) {
   for i in range(0, n, 1):
      x[i] = 100 * r.Rndm(i)
      y[i] = 200 * r.Rndm(i)
      c[i] = 300 * r.Rndm(i)
      s[i] = 400 * r.Rndm(i)
      
   
   global scatter
   scatter = TScatter(n, x, y, c, s)
   scatter.SetMarkerStyle(20)
   scatter.SetMarkerColor(kRed)
   scatter.SetTitle("Scatter plot;X;Y")
   scatter.Draw("A")
   


if __name__ == "__main__":
   scatter()
