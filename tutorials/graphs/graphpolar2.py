## \file
## \ingroup tutorial_graphs
## \notebook
##
## Creates and draws a polar graph with PI axis.
##
## \macro_image
## \macro_code
##
## \author Olivier Couet
## \translator P. P.


import ROOT
import ctypes

#classes
TGraphPolar = ROOT.TGraphPolar
TMath = ROOT.TMath

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
def graphpolar2() :

   global CPol
   CPol = TCanvas("CPol", "TGraphPolar Example", 500, 500)
   
   theta = [ 0. ] * 8 # Double_t*
   radius = [ 0. ] * 8 # Double_t*
   etheta = [ 0. ] * 8 # Double_t*
   eradius = [ 0. ] * 8 # Double_t*
   
   #   for (int i = 0; i < 8; i++) {
   for i in range(0, 8, 1):
      theta[i] = (i + 1) * (TMath.Pi() / 4.)
      radius[i] = (i + 1) * 0.05
      etheta[i] = TMath.Pi() / 8.
      eradius[i] = 0.05
   
   #to c-types
   theta = to_c( theta )
   radius = to_c( radius )
   etheta = to_c( etheta )
   eradius = to_c( eradius )
   
   global grP1
   grP1 = TGraphPolar(8, theta, radius, etheta, eradius)
   grP1.SetTitle("")
   
   grP1.SetMarkerStyle(20)
   grP1.SetMarkerSize(2.)
   grP1.SetMarkerColor(4)
   grP1.SetLineColor(2)
   grP1.SetLineWidth(3)

   grP1.Draw("PE")
   
   CPol.Update()
   
   if (grP1.GetPolargram()) :
      grP1.GetPolargram().SetToRadian()
   


if __name__ == "__main__":
   graphpolar2()
