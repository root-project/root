## \file
## \ingroup tutorial_graphs
## \notebook
##
## Draws 2-Dim functions.
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
TF2 = ROOT.TF2
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
def surfaces() :

   global c1
   c1 = TCanvas("c1", "Surfaces Drawing Options", 200, 10, 700, 900)

   global title
   title = TPaveText(.2, 0.96, .8, .995)
   title.AddText("Examples of Surface options")
   title.Draw()
   

   global pad1, pad2
   pad1 = TPad("pad1", "Gouraud shading", 0.03, 0.50, 0.98, 0.95)
   pad2 = TPad("pad2", "Color mesh", 0.03, 0.02, 0.98, 0.48)
   pad1.Draw()
   pad2.Draw()
   #
   # We generate a 2-D function

   global f2
   f2 = TF2("f2", "x**2 + y**2 - x**3 -8*x*y**4", -1, 1.2, -1.5, 1.5)
   f2.SetContour(48)
   f2.SetFillColor(45)
   
   # Draw this function in pad1 with Gouraud shading option
   pad1.cd()
   pad1.SetPhi(-80)
   pad1.SetLogz()
   f2.Draw("surf4")
   
   # Draw this function in pad2 with color mesh option
   pad2.cd()
   pad2.SetTheta(25)
   pad2.SetPhi(-110)
   pad2.SetLogz()
   f2.SetLineWidth(1)
   f2.SetLineColor(5)
   f2.Draw("surf1")
   
   # add axis titles. The titles are set on the intermediate
   # histogram used for visualisation. We must force this histogram
   # to be created, then force the redrawing of the two pads
   pad2.Update()
   f2.GetHistogram().GetXaxis().SetTitle("x title")
   f2.GetHistogram().GetYaxis().SetTitle("y title")
   f2.GetHistogram().GetXaxis().SetTitleOffset(1.4)
   f2.GetHistogram().GetYaxis().SetTitleOffset(1.4)

   pad1.Modified()
   pad2.Modified()
   


if __name__ == "__main__":
   surfaces()
