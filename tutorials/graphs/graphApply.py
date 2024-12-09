## \file
## \ingroup tutorial_graphs
## \notebook
##
## A script to demonstrate the functionality of TGraph.Apply() method.
## TGraph.Apply applies a function `f` to all data TGraph points.
## `f` may be a 1-D function TF1 or 2-d function TF2.
## The Y values of the graph are replaced by the new values obtained by computation using
## the function `f`.
##
## \macro_image
## \macro_code
##
## \author Miro Helbich
## \translator P. P.


import ROOT
import ctypes

#classes
TGraphErrors = ROOT.TGraphErrors
TGraphAsymmErrors = ROOT.TGraphAsymmErrors
TF2 = ROOT.TF2
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
def graphApply() :

   npoints = 3

   xaxis = [ 1., 2., 3. ]
   yaxis = [ 10., 20., 30. ]
   errorx = [ 0.5, 0.5, 0.5 ]
   errory = [ 5., 5., 5. ]
   
   exl = [ 0.5, 0.5, 0.5 ]
   exh = [ 0.5, 0.5, 0.5 ]
   eyl = [ 5., 5., 5. ]
   eyh = [ 5., 5., 5. ]
   #to C-types
   xaxis = to_c( xaxis )
   yaxis = to_c( yaxis )
   errorx = to_c( errorx )
   errory = to_c( errory )
   
   exl = to_c( exl )
   exh = to_c( exh )
   eyl = to_c( eyl )
   eyh = to_c( eyh )
   
   global gr1, gr2, gr3, ff 
   gr1 = TGraph(npoints, xaxis, yaxis)
   gr2 = TGraphErrors(npoints, xaxis, yaxis, errorx, errory)
   gr3 = TGraphAsymmErrors(npoints, xaxis, yaxis, exl, exh, eyl, eyh)
   ff = TF2("ff", "-1./y")
   

   global c1
   c1 = TCanvas("c1", "c1")
   c1.Divide(2, 3)
   
   # TGraph
   c1.cd(1)
   gr1.DrawClone("A*")
   c1.cd(2)
   gr1.Apply(ff)
   gr1.Draw("A*")
   
   # TGraphErrors
   c1.cd(3)
   gr2.DrawClone("A*")
   c1.cd(4)
   gr2.Apply(ff)
   gr2.Draw("A*")
   
   # TGraphAsymmErrors
   c1.cd(5)
   gr3.DrawClone("A*")
   c1.cd(6)
   gr3.Apply(ff)
   gr3.Draw("A*")
   


if __name__ == "__main__":
   graphApply()
