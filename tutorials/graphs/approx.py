## \file
## \ingroup tutorial_graphs
## \notebook -js
## 
## This script shows how to test an interpolation function 
## with approximations.
##
## \macro_image
## \macro_code
##
## \author Christian Stratowa, Vienna, Austria.
## \translator P. P.


import ROOT
import ctypes

#classes
TCanvas = ROOT.TCanvas
TGraphSmooth = ROOT.TGraphSmooth

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
char = ROOT.char

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


#variables
vC1 = TCanvas()
grxy, grin, grout = [ TGraph() for _ in range(3) ]

# void
def DrawSmooth(pad : Int_t, title : char, xt : char, yt : char) :

   global vC1
   vC1.cd(pad)

   global vFrame
   vFrame = gPad.DrawFrame(0, 0, 15, 150)

   vFrame.SetTitle(title)
   vFrame.SetTitleSize(0.2)
   vFrame.SetXTitle(xt)
   vFrame.SetYTitle(yt)

   global grxy
   grxy.SetMarkerColor(kBlue)
   grxy.SetMarkerStyle(21)
   grxy.SetMarkerSize(0.5)
   grxy.Draw("P")

   global grin
   grin.SetMarkerColor(kRed)
   grin.SetMarkerStyle(5)
   grin.SetMarkerSize(0.7)
   grin.Draw("P")

   global grout
   grout.DrawClone("LP")
   

# void
def approx() :

   # Test data (square)
   n = 11
   x = [ 1, 2, 3, 4, 5, 6, 6, 6, 8, 9, 10 ]
   y = [ 1, 4, 9, 16, 25, 25, 36, 49, 64, 81, 100 ]
   n = len(x)
   n = len(y)
   # To Python-types
   x = to_c( x )
   y = to_c( y )
   
   global grxy
   grxy = TGraph(n, x, y)
   
   # X values, for which y values should be interpolated
   nout = 14
   xout = [ 1.2, 1.7, 2.5, 3.2, 4.4, 5.2, 5.7, 6.5, 7.6, 8.3, 9.7, 10.4, 11.3, 13 ]
   # To C-types
   xout = to_c( xout )
   
   # yreate Canvas
   global vC1
   vC1 = TCanvas("vC1", "square", 200, 10, 700, 700)
   vC1.Divide(2, 2)
   
   # Initialize graph with data
   global grin
   grin = TGraph(n, x, y)

   # Interpolate at equidistant points (use mean for tied x-values)
   global gs, grout
   gs = TGraphSmooth("normal")
   grout = gs.Approx(grin, "linear")

   DrawSmooth(1, "Approx: ties = mean", "X-axis", "Y-axis")
   
   # Re-initialize graph with data
   # (since graph points were set to unique vales)
   grin = TGraph(n, x, y)

   # Interpolate at given points xout
   grout = gs.Approx(grin, "linear", 14, xout, 0, 130)

   DrawSmooth(2, "Approx: ties = mean", "", "")
   
   # Print output variables for given values xout
   global vNout
   vNout = grout.GetN()
   vXout, vYout = c_double(), c_double() 
   #   for (Int_t k = 0; k < vNout; k++) {
   for k in range(0, vNout, 1):
      grout.GetPoint(k, vXout, vYout)
      #print(f"k= " , k , "  vXout[k]= " , vXout.value , "  vYout[k]= " , vYout.value)
      print(f"k= {k:.1f}  vXout[k]= {vXout.value:.1f}  vYout[k]= {vYout.value:.1f}")
      
   
   # R-initialize graph with data
   grin = TGraph(n, x, y)

   # Interpolate at equidistant points (use min for tied x-values).
   # _grout = gs.Approx(grin,"linear", 50, 0, 0, 0, 1, 0, "min")_
   grout = gs.Approx(grin, "constant", 50, 0, 0, 0, 1, 0.5, "min")

   DrawSmooth(3, "Approx: ties = min", "", "")
   
   # Re-initialize graph with data
   grin = TGraph(n, x, y)

   # Interpolate at equidistant points (use max for tied x-values)
   grout = gs.Approx(grin, "linear", 14, xout, 0, 0, 2, 0, "max")
   DrawSmooth(4, "Approx: ties = max", "", "")
   
   # Cleanup
   del gs
   


if __name__ == "__main__":
   approx()
