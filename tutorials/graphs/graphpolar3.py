## \file
## \ingroup tutorial_graphs
## \notebook
##
## This script creates and draws a polar graph with PI axis using a TF1-class.
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
TMath = ROOT.TMath
TF1 = ROOT.TF1
TGraphPolar = ROOT.TGraphPolar

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
def graphpolar3() :

   global CPol
   CPol = TCanvas("CPol", "TGraphPolar Examples", 500, 500)
   
   rmin = 0. # Double_t
   rmax = TMath.Pi() * 2
   r = [ Double_t() for _ in range(1000) ]
   theta = [ Double_t() for _ in range(1000) ]
   

   global fp1
   fp1 = TF1("fplot", "cos(x)", rmin, rmax)
   #   for (Int_t ipt = 0; ipt < 1000; ipt++) {
   for ipt in range(0, 1000, 1):
      r[ipt] = ipt * (rmax - rmin) / 1000 + rmin
      theta[ipt] = fp1.Eval(r[ipt])
      
   

   global grP1
   r = to_c( r )
   theta = to_c( theta )
   grP1 = TGraphPolar(1000, r, theta)
   grP1.SetTitle("")
   grP1.SetLineColor(2)
   grP1.Draw("AOL")
   


if __name__ == "__main__":
   graphpolar3()
