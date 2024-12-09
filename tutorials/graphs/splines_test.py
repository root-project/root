## \file
## \ingroup tutorial_graphs
## \notebook
## Examples on how to use the spline-related classes of ROOT.
## TSpline3 and TSpline5 will be used to form splines of 
## third and fifth polynomical order.
## Enjoy the plot; it is pretty dynamic.
##
## \macro_image
## \macro_code
##
## \author Federico Carminati
## \translator P. P.


import ROOT
import ctypes

#classes
TSpline3 = ROOT.TSpline3
TSpline5 = ROOT.TSpline5

TCanvas = ROOT.TCanvas
TF1 = ROOT.TF1
TMath = ROOT.TMath


TH1F = ROOT.TH1F
TGraph = ROOT.TGraph
TLatex = ROOT.TLatex
TLine = ROOT.TLine

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
gSystem = ROOT.gSystem
gStyle = ROOT.gStyle
gPad = ROOT.gPad
gRandom = ROOT.gRandom
gBenchmark = ROOT.gBenchmark
gROOT = ROOT.gROOT


# void
def splines_test(np : Int_t = 23, a : Double_t = -0.5, b : Double_t = 31) :

   ## array of points
   xx = [ 0.] ; yy = [ 0. ] # Double_t *
   
   global spline3, spline5
   spline3 = [ TSpline3() ] # nullptr
   spline5 = [ TSpline5() ] # nullptr
   #line5, line3 = [ TLine() for _ in range(2) ]
   #text5, text3, textn = [ TText() for _ in range(3) ]
   text = " "*20 # char
   power = 0.75 # Double_t
   
   # Define the original function

   global f
   f = TF1("f", "sin(x)*sin(x/10)", a - 0.05 * (b - a), b + 0.05 * (b - a))

   # Draw function
   f.Draw("lc")
   
   # Create text and legend
   global xx1, yy1, xx2, yy2, dx, dy
   xx1, yy1, xx2, yy2, dx, dy = [ c_double() for _ in range(6) ]

   gPad.Update()
   gPad.GetRangeAxis(xx1, yy1, xx2, yy2)

   #to Python-types
   xx1 = xx1.value
   yy1 = yy1.value
   xx2 = xx2.value
   yy2 = yy2.value
   dx = dx.value
   dy = dy.value

   #Begining operations.
   dx = xx2 - xx1
   dy = yy2 - yy1


   global line5
   line5 = TLine(xx1 + dx * 0.3, yy1 + dy * 1.02, xx1 + dx * 0.38, yy1 + dy * 1.02)
   line5.SetLineColor(kRed)
   line5.SetLineWidth(2)

   global text5
   text5 = TText(xx1 + dx * 0.4, yy1 + dy * 1.03, "quintic spline")
   text5.SetTextAlign(12)
   text5.SetTextSize(0.04)

   global line3
   line3 = TLine(xx1 + dx * 0.67, yy1 + dy * 1.02, xx1 + dx * 0.75, yy1 + dy * 1.02)
   line3.SetLineColor(kGreen)
   line3.SetLineWidth(2)

   global text3
   text3 = TText(xx1 + dx * 0.77, yy1 + dy * 1.03, "third spline")
   text3.SetTextAlign(12)
   text3.SetTextSize(0.04)

   global textn
   textn = TText(xx1 + dx * 0.8, yy1 + dy * 0.91, " ")
   textn.SetTextAlign(12)
   textn.SetTextSize(0.04)
   textn.Draw()
   
   # Draw legends
   line5.Draw()
   text5.Draw()
   line3.Draw()
   text3.Draw()
   
   #   for (Int_t nnp = 2; nnp <= np; ++nnp) {
   for nnp in range(2, np + 1, 1):
      
      # Calculate the knots
      if (xx) :
         del xx
      xx = [ Double_t() ]*nnp
      if (yy) :
         del yy
      yy = [ Double_t() ]*nnp

      #      for (Int_t i = 0; i < nnp; ++i) {
      for i in range(0, nnp, 1):

         xx[i] = a + (b - a) * TMath.Power(i / Double_t(nnp - 1), power)
         yy[i] = f.Eval(xx[i])
         
      
      # Evaluate fifth spline coefficients
      eps = (b - a) * 1.e-5

      if (spline5) :
         del spline5

      c_xx = to_c( xx )
      #global spline5 # already global
      spline5 = TSpline5(
                   "Test",
                   c_xx,
                   f,
                   nnp,
                   "b1e1b2e2",
                   f.Derivative(a),
                   f.Derivative(b),
                   (f.Derivative(a + eps) - f.Derivative(a)) / eps,
                   (f.Derivative(b) - f.Derivative(b - eps)) / eps
      )

      
      spline5.SetLineColor(kRed)
      spline5.SetLineWidth(3)
      
      # Draw the quintic spline
      spline5.Draw("lcsame")
      
      # Evaluate third spline coefficients
      #global spline3 # already global
      if (spline3) :
         del spline3

      c_xx = to_c( xx ) 
      c_yy = to_c( yy ) 
      spline3 = TSpline3(
             "Test",
             c_xx,
             c_yy,
             nnp,
             "b1e1",
             f.Derivative(a),
             f.Derivative(b)
      )

      
      spline3.SetLineColor(kGreen)
      spline3.SetLineWidth(3)
      spline3.SetMarkerColor(kBlue)
      spline3.SetMarkerStyle(20)
      spline3.SetMarkerSize(1.5)
      
      # Draw the third spline
      spline3.Draw("lcpsame")
      
      sprintf(text, "%3d knots", nnp)
      textn.SetTitle(text)
      gPad.Update()
      
      # Let it wait till spline-process finishes. Just let it be.
      gSystem.Sleep(500)
      
   


if __name__ == "__main__":
   splines_test()
