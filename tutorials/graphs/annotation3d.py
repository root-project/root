## \file
## \ingroup tutorial_graphs
## \notebook
##
## This example show how to put some annotation on a 3D plot using 3D
## polylines. It also demonstrates how the axis labels can be modified.
## It was created for the book:
## [Statistical Methods for Data Analysis in Particle Physics](http:#www.springer.com/la/book/9783319201757)
## \macro_image
## \macro_code
##
## \author Luca Lista, Olivier Couet
## \translator P. P.


import ROOT
import ctypes

#classes
TCanvas = ROOT.TCanvas
TF2 = ROOT.TF2
TPolyLine3D = ROOT.TPolyLine3D
TAnnotation = ROOT.TAnnotation

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
kAzure = ROOT.kAzure
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
def annotation3d() :

   global c
   c = TCanvas("c", "c", 600, 600)
   c.SetTheta(30)
   c.SetPhi(50)

   gStyle.SetOptStat(0)
   gStyle.SetHistTopMargin(0)
   gStyle.SetOptTitle(False)
   
   # Define and draw a surface

   global f
   f = TF2("f", "[0]*cos(x)*cos(y)", -1, 1, -1, 1)
   f.SetParameter(0, 1)

   #scale factor
   s = 1./f.Integral(-1, 1, -1, 1)
   f.SetParameter(0, s)

   f.SetNpx(50)
   f.SetNpy(50)
   
   f.GetXaxis().SetTitle("x")
   f.GetXaxis().SetTitleOffset(1.4)
   f.GetXaxis().SetTitleSize(0.04)
   f.GetXaxis().CenterTitle()
   f.GetXaxis().SetNdivisions(505)
   f.GetXaxis().SetTitleOffset(1.3)
   f.GetXaxis().SetLabelSize(0.03)
   f.GetXaxis().ChangeLabelByValue(-0.5,-1,-1,-1,kRed,-1,"X_{0}")
   
   f.GetYaxis().SetTitle("y")
   f.GetYaxis().CenterTitle()
   f.GetYaxis().SetTitleOffset(1.4)
   f.GetYaxis().SetTitleSize(0.04)
   f.GetYaxis().SetTitleOffset(1.3)
   f.GetYaxis().SetNdivisions(505)
   f.GetYaxis().SetLabelSize(0.03)
   
   f.GetZaxis().SetTitle("dP/dx")
   f.GetZaxis().CenterTitle()
   f.GetZaxis().SetTitleOffset(1.3)
   f.GetZaxis().SetNdivisions(505)
   f.GetZaxis().SetTitleSize(0.04)
   f.GetZaxis().SetLabelSize(0.03)
   
   f.SetLineWidth(1)
   f.SetLineColorAlpha(kAzure-2, 0.3)
   
   f.Draw("surf1 fb")
   
   # Lines for 3D annotation
   x = [ -0.500, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.500 ]
   y = [ -0.985, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 0.985 ]
   z = [ 0. ] * len(x)
   #for (i = 0; i < 11; ++i) 
   for i in range(0, 11, 1):
      z[i] = s*cos(x[i])*cos(y[i])

   #to C-types
   x = to_c( x )
   y = to_c( y )
   z = to_c( z )

   global g2
   g2 = TPolyLine3D(11, x, y, z)
   
   xx = [ -0.5, -0.5 ]
   yy = [ -0.985, -0.985 ]
   zz = [ 0.11, s*cos(-0.5)*cos(-0.985) ]
   #to C-types
   xx = to_c( xx )
   yy = to_c( yy )
   zz = to_c( zz )

   global l2
   l2 = TPolyLine3D(2, xx, yy, zz)
   
   g2.SetLineColor(kRed)
   g2.SetLineWidth(3)
   g2.Draw()
   
   l2.SetLineColor(kRed)
   l2.SetLineStyle(2)
   l2.SetLineWidth(1)
   l2.Draw()
   
   # Draw text Annotations

   global txt
   txt = TAnnotation(-0.45, -0.2, 0.3, "f(y,x_{0})")
   txt.SetTextFont(42)
   txt.SetTextColor(kRed)
   txt.Draw()
   

   global txt1
   txt1 = TAnnotation(0.5, 0.5, 0.3, "f(x,y)")
   txt1.SetTextColor(kBlue)
   txt1.SetTextFont(42)
   txt1.Draw()
   


if __name__ == "__main__":
   annotation3d()
