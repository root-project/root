## \file
## \ingroup tutorial_graphs
## \notebook
##
## Draws a graph with text attached to each point.
## The text is drawn in a TExec-object method which is attached to the TGraph-object;
## therefore, if the a graph's point is
## moved interactively, the text will be automatically updated.
##
## \macro_image
## \macro_code
##
## \author Olivier Couet
## \translator P. P.


import ROOT
import ctypes

#classes
TExec = ROOT.TExec
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
def Form( string, *args):
   return string % args

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
def graphtext() :

   global c
   c = TCanvas("c", "A Simple Graph Example with Text", 700, 500)
   c.SetGrid()
   
   n = 10

   global gr
   gr = TGraph(n)
   gr.SetTitle("A Simple Graph Example with Text")
   gr.SetMarkerStyle(20)

   global ex
   command = """
   TPython::Eval( \"drawtext()\" );
   """ 
   ex = TExec("ex", command)
   gr.GetListOfFunctions().Add(ex)
   
   x, y = 0., 0. # Double_t
   #   for (Int_t i = 0; i < n; i++) {
   for i in range(0, n, 1):
      x = i * 0.1
      y = 10 * sin(x + 0.2)
      gr.SetPoint(i, x, y)
      
   gr.Draw("ALP")
   

# void
def drawtext() :

   i, n = 0, 0 # Int_t()
   c_x, c_y = c_double(), c_double() #

   global l
   l = TLatex()
   
   l.SetTextSize(0.025)
   l.SetTextFont(42)
   l.SetTextAlign(21)
   l.SetTextColor(kBlue)
   
   g = gPad.GetListOfPrimitives().FindObject("Graph") # TGraph
   n = g.GetN()
   
   #   for (i = 0; i < n; i++) {
   for i in range(0, n, 1):

      g.GetPoint(i, c_x, c_y)
      x = c_x.value
      y = c_y.value

      l.PaintText(x, y + 0.2, Form("(%4.2f,%4.2f)", x, y))
      
   


if __name__ == "__main__":
   graphtext()
