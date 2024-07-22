## \file
## \ingroup tutorial_graphs
## \notebook
##
## Draws a simple graph.
##
## \macro_image
## \macro_code
##
## \author Rene Brun
## \translator P. P.


import ROOT
import ctypes

#classes
TGraph = ROOT.TGraph
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
   print( string % args , end = "")
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
def graph() :

   global c1
   c1 = TCanvas("c1", "A Simple Graph Example", 200, 10, 700, 500)
   c1.SetGrid()
   
   n = 20
   x, y = [ to_c( [0.]*n ) for _ in range(2) ]
   #   for (Int_t i = 0; i < n; i++) {
   for i in range(0, n, 1):
      x[i] = i * 0.1
      y[i] = 10 * sin(x[i] + 0.2)
      printf(" i %i %f %f \n", i, x[i], y[i])
      

   global gr
   gr = TGraph(n, x, y)

   #Setting-up.
   gr.SetLineColor(2)
   gr.SetLineWidth(4)
   gr.SetMarkerColor(4)
   gr.SetMarkerStyle(21)
   gr.SetTitle("a simple graph")
   gr.GetXaxis().SetTitle("X title")
   gr.GetYaxis().SetTitle("Y title")
   
   #Draw on canvas.
   gr.Draw("ACP")
   
   # TCanvas::Update() draws the frame, after which one can change it
   c1.Update()
   c1.GetFrame().SetBorderSize(12)
   c1.Modified()
   


if __name__ == "__main__":
   graph()
