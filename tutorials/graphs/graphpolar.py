## \file
## \ingroup tutorial_graphs
## \notebook
##
## Creates and draws a polar graph.
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
def graphpolar() :
   # Illustrates how to use TGraphPolar
   
   global CPol
   CPol = TCanvas("CPol", "TGraphPolar Examples", 1200, 600)
   CPol.Divide(2, 1)
   CPol.cd(1)
   
   xmin = 0
   xmax = TMath.Pi() * 2
   
   x = [ Double_t() for _ in range(1000) ]
   y = [ Double_t() for _ in range(1000) ]
   xval1 = [ Double_t() for _ in range(20) ]
   yval1 = [ Double_t() for _ in range(20) ]
   

   global fplot
   fplot = TF1("fplot", "cos(2*x)*cos(20*x)", xmin, xmax)
   
   #   for (Int_t ipt = 0; ipt < 1000; ipt++) {
   for ipt in range(0, 1000, 1):
      x[ipt] = ipt * (xmax - xmin) / 1000 + xmin
      y[ipt] = fplot.Eval(x[ipt])
      
   x = to_c( x ) 
   y = to_c( y ) 
   global grP
   grP = TGraphPolar(1000, x, y)
   grP.SetLineColor(2)
   grP.SetLineWidth(2)
   grP.SetFillStyle(3012)
   grP.SetFillColor(2)
   grP.Draw("AFL")
   
   #   for (Int_t ipt = 0; ipt < 20; ipt++) {
   for ipt in range(0, 20, 1):
      xval1[ipt] = x[1000 // 20 * ipt]
      yval1[ipt] = y[1000 // 20 * ipt]
      

   xval1 = to_c( xval1 )
   yval1 = to_c( yval1 )
   global grP1
   grP1 = TGraphPolar(20, xval1, yval1)
   grP1.SetMarkerStyle(29)
   grP1.SetMarkerSize(2)
   grP1.SetMarkerColor(4)
   grP1.SetLineColor(4)
   grP1.Draw("CP")
   
   # Update, otherwise GetPolargram returns 0
   CPol.Update()
   if grP1.GetPolargram():
      grP1.GetPolargram().SetTextColor(8)
      grP1.GetPolargram().SetRangePolar(-TMath.Pi(), TMath.Pi())
      grP1.GetPolargram().SetNdivPolar(703)
      grP1.GetPolargram().SetToRadian()
      
   
   CPol.cd(2)
   x2 = [ Double_t() for _ in range(30) ]
   y2 = [ Double_t() for _ in range(30) ]
   ex = [ Double_t() for _ in range(30) ]
   ey = [ Double_t() for _ in range(30) ]
   #   for (Int_t ipt = 0; ipt < 30; ipt++) {
   for ipt in range(0, 30, 1):
      x2[ipt] = x[1000 // 30 * ipt]
      y2[ipt] = 1.2 + 0.4 * sin(TMath.Pi() * 2 * ipt / 30)
      ex[ipt] = 0.2 + 0.1 * cos(2 * TMath.Pi() / 30 * ipt)
      ey[ipt] = 0.2
      
   
   x2 = to_c( x2 )
   y2 = to_c( y2 )
   ex = to_c( ex )
   ey = to_c( ey )
   global grPE
   grPE = TGraphPolar(30, x2, y2, ex, ey)
   grPE.SetMarkerStyle(22)
   grPE.SetMarkerSize(1.5)
   grPE.SetMarkerColor(5)
   grPE.SetLineColor(6)
   grPE.SetLineWidth(2)
   grPE.Draw("EP")
   # Update, otherwise GetPolargram returns 0
   CPol.Update()
   
   if grPE.GetPolargram():
      grPE.GetPolargram().SetTextSize(0.03)
      grPE.GetPolargram().SetTwoPi()
      grPE.GetPolargram().SetToRadian()
      
   


if __name__ == "__main__":
   graphpolar()
