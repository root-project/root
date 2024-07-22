## \file
## \ingroup tutorial_graphs
## \notebook
##
## Defines a time offset like 2003, January 1st.
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
TDatime = ROOT.TDatime
TRandom = ROOT.TRandom
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
def timeonaxis2() :

   global ct2
   ct2 = TCanvas("ct2", "ct2", 10, 10, 700, 500)
   
   global T0
   T0 = TDatime(2003, 1, 1, 0, 0, 0)
   X0 = T0.Convert()

   gStyle.SetTimeOffset(X0)
   
   # Define the lowest histogram limit as 2002, September 23rd
   global T1
   T1 = TDatime(2002, 9, 23, 0, 0, 0)
   X1 = T1.Convert() - X0
   
   # Define the highest histogram limit as 2003, March 7th
   global T2
   T2 = TDatime(2003, 3, 7, 0, 0, 0)
   X2 = T2.Convert(1) - X0
   
   global h1
   h1 = TH1F("h1", "test", 100, X1, X2)
   
   global r
   r = TRandom()
   #   for (Int_t i = 0; i < 30000; i++) {
   for i in range(0, 30000, 1):
      noise = r.Gaus(0.5 * (X1 + X2), 0.1 * (X2 - X1))
      h1.Fill(noise)
      
   
   h1.GetXaxis().SetTimeDisplay(1)
   h1.GetXaxis().SetLabelSize(0.03)
   h1.GetXaxis().SetTimeFormat("%Y/%m/%d")

   h1.Draw()

   return ct2 # TCanvas
   


if __name__ == "__main__":
   timeonaxis2()
