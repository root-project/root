## \file
## \ingroup tutorial_graPads
##
## This script demonstrates how to use the highlight mode on graph.
##
## \macro_code
##
## \date March 2018
## \author Jan Musinsky
## \translator P. P.


import ROOT
import ctypes

#classes
TCanvas = ROOT.TCanvas
TString = ROOT.TString 
TVirtualPad = ROOT.TVirtualPad
TObject = ROOT.TObject
TPyDispatcher = ROOT.TPyDispatcher 
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
def TString_Format( string, *args):
   return TString( string % args )
TString.Format = TString_Format

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



#Deprecated.
#l = TList() nullptr
l = [] # nullptr

# void
def HighlightHisto(pad : TVirtualPad, obj : TObject, ihp : Int_t ) :

   global Pad
   Pad = pad.FindObject("Pad") # TVirtualPad
   if ( not Pad) :
      return

   # after highlight disabled
   if (ihp == -1) :
      Pad.Clear()
      return
      
   
   #Deprecated: if l and l.At(ihp):
   if l and l[ihp]:
      Pad.cd()
      #Deprecated: l.At(ihp).Draw()
      l[ihp].Draw()
      gPad.Update()
      
   

# void
def hlGraph1() :


   global Canvas
   Canvas = TCanvas("Canvas", "Canvas", 0, 0, 700, 500)
   PyD_HighlightHisto = TPyDispatcher( HighlightHisto )
   Canvas.Connect("Highlighted(TVirtualPad*,TObject*,Int_t,Int_t))",
                  "TPyDispatcher",
                  PyD_HighlightHisto,
                  "Dispatch(TVirtualPad*,TObject*,Int_t))"
                  )
   #Canvas.HighlightConnect("HighlightHisto(TVirtualPad*,TObject*,Int_t,Int_t)")
   n = 500
   x = [ Double_t() for _ in range(n) ]
   y = [ Double_t() for _ in range(n) ]

   #Deprecated:
   #l = TList()
   global l
   l = [] 
   
   global h, h_list
   h_list = []
   #   for (Int_t i = 0; i < n; i++) {
   for i in range(0, n, 1):

      h = TH1F("h_%03d"%i, "", 100, -3.0, 3.0)
      h.FillRandom("gaus", 1000)
      h.Fit("gaus", "Q")
      h.SetMaximum(250.0); # for n > 200
      #Deprecated: l.Add(h)
      l.append(h)
      x[i] = i
      y[i] = h.GetFunction("gaus").GetParameter(2)
      
      h_list.append( h )
      
   

   x = to_c(x)
   y = to_c(y) 
   global g
   g = TGraph(n, x, y)
   g.SetMarkerStyle(6)
   g.Draw("AP")
   

   global Pad
   Pad = TPad("Pad", "Pad", 0.3, 0.4, 1.0, 1.0)
   Pad.SetFillColor(kBlue - 10)
   Pad.Draw()
   Pad.cd()

   global info
   info = TText(0.5, 0.5, "please move the mouse over the graPad")
   info.SetTextAlign(22)
   info.Draw()
   Canvas.cd()
   
   g.SetHighlight()
   


if __name__ == "__main__":
   hlGraph1()
