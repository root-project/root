## \file
## \ingroup tutorial_graphs
## \notebook
##
## Draws and fits a TGraph2DErrors-objects.
##
## \macro_image
## \macro_code
##
## \author Olivier Couet
## \translator P. P.


import ROOT
import ctypes
TCanvas = ROOT.TCanvas 
TF2 = ROOT.TF2 
TGraph2DErrors = ROOT.TGraph2DErrors 
TMath = ROOT.TMath 
TRandom = ROOT.TRandom 
TStyle = ROOT.TStyle 

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
def graph2derrorsfit() :

   global c1
   c1 = TCanvas("c1")
   
   rnd, x, y, z, ex, ey, ez = [ Double_t() for _ in range(7) ]
   e = 0.3 # Double_t
   nd = 500 # Int_t
   global r
   r = TRandom()
   

   global f2
   f2 = TF2("f2", "1000*(([0]*sin(x)/x)*([1]*sin(y)/y))+200", -6, 6, -6, 6)
   f2.SetParameters(1, 1)

   global dte
   dte = TGraph2DErrors(nd)
   
   # Fill the 2D graph
   zmax = 0
   #   for (Int_t i = 0; i < nd; i++) {
   for i in range(0, nd, 1):

      #Getting random values from f2.
      x, y = map( c_double, [x, y] )
      f2.GetRandom2(x, y)
      x, y = map( lambda c_obj: c_obj.value , [x, y] )

      # Generate a random number in [-e,e]
      rnd = r.Uniform(-e, e); 
      z = f2.Eval(x, y) * (1 + rnd)
      if (z > zmax):
         zmax = z
      dte.SetPoint(i, x, y, z)
      ex = 0.05 * r.Rndm()
      ey = 0.05 * r.Rndm()
      ez = TMath.Abs(z * rnd)
      dte.SetPointError(i, ex, ey, ez)
      
   #Setting-up again. 
   f2.SetParameters(0.5, 1.5)

   dte.Fit(f2)

   global fit2
   fit2 = dte.FindObject("f2") # TF2
   fit2.SetTitle("Minuit fit result on the Graph2DErrors points")
   fit2.SetMaximum(zmax)

   gStyle.SetHistTopMargin(0)
   fit2.SetLineColor(1)
   fit2.SetLineWidth(1)


   fit2.Draw("surf1")
   dte.Draw("same p0")
   


if __name__ == "__main__":
   graph2derrorsfit()
