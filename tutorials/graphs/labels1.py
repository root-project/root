## \file
## \ingroup tutorial_graphs
## \notebook
##
## Setting alphanumeric labels in a 1-d histogram.
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
def labels1() :

   i = Int_t()
   nx = 20

   people = ["Jean", "Pierre", "Marie", "Odile", "Sebastien", "Fons", "Rene", "Nicolas", "Xavier", "Greg", "Bjarne", "Anton", "Otto", "Eddy", "Peter", "Pasha", "Philippe", "Suzanne", "Jeff", "Valery"]


   global c1
   c1 = TCanvas("c1", "demo bin labels", 10, 10, 900, 500)
   c1.SetGrid()
   c1.SetBottomMargin(0.15)

   global h
   h = TH1F("h", "test", nx, 0, nx)
   h.SetFillColor(38)

   #   for (i = 0; i < 5000; i++) {
   for i in range(0, 5000, 1):
      h.Fill(gRandom.Gaus(0.5 * nx, 0.2 * nx))

   h.SetStats(0)

   #   for (i = 1; i <= nx; i++) {
   for i in range(1, nx + 1, 1):
      h.GetXaxis().SetBinLabel(i, people[i - 1])

   h.Draw()


   global pt
   pt = TPaveText(0.6, 0.7, 0.98, 0.98, "brNDC")

   pt.SetFillColor(18)
   pt.SetTextAlign(12)
   pt.AddText("Use the axis Context Menu LabelsOption")
   pt.AddText(" \"a\"   to sort by alphabetic order")
   pt.AddText(" \">\"   to sort by decreasing values")
   pt.AddText(" \"<\"   to sort by increasing values")

   pt.Draw()
   


if __name__ == "__main__":
   labels1()
