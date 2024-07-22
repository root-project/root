## \file
## \ingroup tutorial_graphs
## \notebook
##
## Shows how to shade an area between two graphs.
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
def to_py( c_ls ):
   return list( c_ls ) 
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
def graphShade() :

   global c1
   c1 = TCanvas("c1", "A Simple Graph Example", 200, 10, 700, 500)
   
   c1.SetGrid()
   c1.DrawFrame(0, 0, 2.2, 12)
   
   n = 20
   x, y, ymin, ymax = [ [0.]*n for _ in range(4) ]
   i = Int_t() # loop index

   #   for (i = 0; i < n; i++) {
   for i in range(0, n, 1):
      x[i] = 0.1 + i * 0.1
      ymax[i] = 10 * sin(x[i] + 0.2)
      ymin[i] = 8 * sin(x[i] + 0.1)
      y[i] = 9 * sin(x[i] + 0.15)
      
   #to c-types
   x = to_c( x )
   y = to_c( y )
   ymin = to_c( ymin )
   ymax = to_c( ymax )

   global grmin, grmax, gr
   grmin = TGraph(n, x, ymin)
   grmax = TGraph(n, x, ymax)
   gr = TGraph(n, x, y)

   global grshade
   grshade = TGraph(2 * n)
   #to python-types
   x = to_py( x )
   ymin = to_py( ymin )
   ymax = to_py( ymax )

   #   for (i = 0; i < n; i++) {
   for i in range(0, n, 1):

      grshade.SetPoint(i, x[i], ymax[i])
      grshade.SetPoint(n + i, x[n - i - 1], ymin[n - i - 1])
      
   grshade.SetFillStyle(3013)
   grshade.SetFillColor(16)


   #Draw shade with min and max.
   grshade.Draw("f")
   grmin.Draw("l")
   grmax.Draw("l")
  
   #Draw the graph.
   gr.SetLineWidth(4)
   gr.SetMarkerColor(4)
   gr.SetMarkerStyle(21)

   gr.Draw("CP")
   


if __name__ == "__main__":
   graphShade()
