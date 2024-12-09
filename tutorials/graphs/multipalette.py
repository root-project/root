## \file
## \ingroup tutorial_graphs
## \notebook
##
## Draws many color plots using different color palettes.
##
## Since only one palette is active, one need to use `TExec` to be able to
## display many plots using different palettes on the same pad.
##
## When a pad is painted, all its elements are painted in the sequence
## the Draw-method calls(see the difference between Draw- and Paint-methods in the TPad documentation).
## For TExec: it executes its command(s) which, in the following
## example, sets the palette for painting all the objects afterwards.
## Now, if in the next pad another TExec-object changes the palette, it doesn’t affect the
## previous pad which was already painted, but it will affect the current one and
## those painted later.
##
## The following macro illustrate this feature.
##
## \macro_image
## \macro_code
##
## \author Olivier Couet
## \translator P. P.


import ROOT
import ctypes
TCanvas = ROOT.TCanvas 
TColor = ROOT.TColor 
TExec = ROOT.TExec 
TF2 = ROOT.TF2 
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
kBlack = ROOT.kBlack
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
def Pal1() :
   colors = [ Int_t() for i in range(50) ]
   initialized = False
   
   Red = [ 1.00, 0.00, 0.00 ]
   Green = [ 0.00, 1.00, 0.00 ]
   Blue = [ 1.00, 0.00, 1.00 ]
   Length = [ 0.00, 0.50, 1.00 ]
   #to C-types
   Red = to_c( Red )
   Green = to_c( Green )
   Blue = to_c( Blue )
   Length = to_c( Length )
   
   if not initialized:
      FI = TColor.CreateGradientColorTable(3, Length, Red, Green, Blue, 50)
      #      for (i = 0; i < 50; i++) {
      for i in range(0, 50, 1):
         colors[i] = FI + i
      initialized = True
      return
      
   global gStyle
   gStyle.SetPalette(50, colors)
   

# void
def Pal2() :
   colors = [ Int_t() for _ in range(50) ]
   initialized = False
   
   Red = [ 1.00, 0.50, 0.00 ]
   Green = [ 0.50, 0.00, 1.00 ]
   Blue = [ 1.00, 0.00, 0.50 ]
   Length = [ 0.00, 0.50, 1.00 ]
   #to C-types
   Red = to_c( Red )
   Green = to_c( Green )
   Blue = to_c( Blue )
   Length = to_c( Length )
   
   if not initialized:
      FI = TColor.CreateGradientColorTable(3, Length, Red, Green, Blue, 50)
      #      for (i = 0; i < 50; i++) {
      for i in range(0, 50, 1):
         colors[i] = FI + i
      initialized = True
      return
      
   global gStyle
   gStyle.SetPalette(50, colors)
   

# void
def multipalette() :

   global c3
   c3 = TCanvas("c3", "C3", 0, 0, 600, 400)
   c3.Divide(2, 1)

   global f3
   f3 = TF2("f3", "0.1+(1-(x-2)*(x-2))*(1-(y-2)*(y-2))", 1, 3, 1, 3)
   f3.SetLineWidth(1)
   f3.SetLineColor(kBlack)

   c3.cd(1)
   f3.Draw("surf1")
   global ex1
   #ex1 = TExec("ex1", "Pal1();")
   execution1 = """
   TPython::Eval( \"Pal1()\");
   """
   ex1 = TExec("ex1", execution1) 
   ex1.Draw()
   f3.Draw("surf1 same")
   
   c3.cd(2)
   f3.Draw("surf1")
   global ex2
   #ex2 = TExec("ex2", "Pal2();")
   execution2 = """
   TPython::Eval( \"Pal2()\");
   """
   ex2 = TExec("ex2", execution2) 
   ex2.Draw()
   f3.Draw("surf1 same")
   


if __name__ == "__main__":
   multipalette()
