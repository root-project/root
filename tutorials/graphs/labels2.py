## \file
## \ingroup tutorial_graphs
## \notebook
##
## This script shows how to set-up alphanumeric labels on a TCanvas-object.
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
TH2F = ROOT.TH2F
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
def labels2() :

   i = Int_t()
   nx = 12
   ny = 20

   month = [ "January",
          "February",
          "March",
          "April",
          "May",
          "June",
          "July",
          "August",
          "September",
          "October",
          "November",
          "December" ]

   people = ["Jean",
          "Pierre",
          "Marie",
          "Odile",
          "Sebastien",
          "Fons",
          "Rene",
          "Nicolas",
          "Xavier",
          "Greg",
          "Bjarne",
          "Anton",
          "Otto",
          "Eddy",
          "Peter",
          "Pasha",
          "Philippe",
          "Suzanne",
          "Jeff",
          "Valery"]


   global c1
   c1 = TCanvas("c1", "demo bin labels", 10, 10, 800, 800)
   c1.SetGrid()
   c1.SetLeftMargin(0.15)
   c1.SetBottomMargin(0.15)

   global h
   h = TH2F("h", "test", nx, 0, nx, ny, 0, ny)
   #   for (i = 0; i < 5000; i++) {
   for i in range(0, 5000, 1):
      h.Fill(gRandom.Gaus(0.5 * nx, 0.2 * nx), gRandom.Gaus(0.5 * ny, 0.2 * ny))
      
   h.SetStats(0)
   #   for (i = 1; i <= nx; i++) {
   for i in range(1, nx + 1, 1):
      h.GetXaxis().SetBinLabel(i, month[i - 1])
   #   for (i = 1; i <= ny; i++) {
   for i in range(1, ny + 1, 1):
      h.GetYaxis().SetBinLabel(i, people[i - 1])
   h.Draw("text")
   

   global pt
   pt = TPaveText(0.6, 0.85, 0.98, 0.98, "brNDC")
   pt.SetFillColor(18)
   pt.SetTextAlign(12)
   pt.AddText("Use the axis Context Menu LabelsOption")
   pt.AddText(" \"a\"   to sort by alphabetic order")
   pt.AddText(" \">\"   to sort by decreasing values")
   pt.AddText(" \"<\"   to sort by increasing values")
   pt.Draw()
   


if __name__ == "__main__":
   labels2()
