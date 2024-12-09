## \file
## \ingroup tutorial_graphs
## \notebook
##
## This script draws a simple graph structure.
## Its layout graph is made by using TGraphStruct-class. Here we create some
## nodes and edges, and then we change a few graphical attributes on some of them.
##
## Note: 
##       TGraphStruct-class comes in libGViz.so library.
##       Use a version of ROOT that has it.
##
## \macro_image
## \macro_code
## \note For this to work, ROOT has to be compiled with gviz ON
## \author Olivier Couet
## \translator P. P.


import ROOT
import ctypes

#classes
#From libGviz :
#TGraphStruct = ROOT.TGraphStruct
#TGraphEdge = ROOT.TGraphEdge

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
kYellow = ROOT.kYellow
kViolet = ROOT.kViolet

#globals
gInterpreter = ROOT.gInterpreter
gSystem = ROOT.gSystem 
gStyle = ROOT.gStyle
gPad = ROOT.gPad
gRandom = ROOT.gRandom
gBenchmark = ROOT.gBenchmark
gROOT = ROOT.gROOT


# TCanvas
def graphstruct() :

   #if not gInterpreter.LoadMacro("TGraphStruct.h"):
   if gSystem.Load("libGviz.so") == -1: 
      #raise RuntimeError("libGViz.so was not installed.", "Please, install it.")
      print("\n >>> libGViz.so was not installed.", "Please, install it.")
      return
       

   global gs
   gs = TGraphStruct()

   # create some nodes and put them in the graph in one go ...
   global n0, n1, n2, n3, n4, n5, n6, n7, n8, n9    
   n0 = gs.AddNode("n0", "Node 0")
   n1 = gs.AddNode("n1", "First node")
   n2 = gs.AddNode("n2", "Second node")
   n3 = gs.AddNode("n3", "Third node")
   n4 = gs.AddNode("n4", "Fourth node")
   n5 = gs.AddNode("n5", "5th node")
   n6 = gs.AddNode("n6", "Node number six")
   n7 = gs.AddNode("n7", "Node 7")
   n8 = gs.AddNode("n8", "Node 8")
   n9 = gs.AddNode("n9", "Node 9")
   
   n4.SetTextSize(0.03)
   n6.SetTextSize(0.03)
   n2.SetTextSize(0.04)
   
   n3.SetTextFont(132)
   
   n0.SetTextColor(kRed)
   
   n9.SetFillColor(kRed - 10)
   n0.SetFillColor(kYellow - 9)
   n7.SetFillColor(kViolet - 9)
   
   # some edges ...
   gs.AddEdge(n0, n1).SetLineColor(kRed)

   global e06
   e06 = gs.AddEdge(n0, n6) # TGraphEdge
   e06.SetLineColor(kRed - 3)
   e06.SetLineWidth(4)

   gs.AddEdge(n1, n7)
   gs.AddEdge(n4, n6)
   gs.AddEdge(n3, n9)
   gs.AddEdge(n6, n8)
   gs.AddEdge(n7, n2)
   gs.AddEdge(n8, n3)
   gs.AddEdge(n2, n3)
   gs.AddEdge(n9, n0)
   gs.AddEdge(n1, n4)
   gs.AddEdge(n1, n6)
   gs.AddEdge(n2, n5)
   gs.AddEdge(n3, n6)
   gs.AddEdge(n4, n5)
   
   global c
   c = TCanvas("c", "c", 800, 600)
   c.SetFillColor(38)

   gs.Draw()
   c.Draw()
   c.Update()

   return c
   


if __name__ == "__main__":
   graphstruct()
