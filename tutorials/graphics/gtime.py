## \file
## \ingroup tutorial_graphics
##
## An example of a graph of data moving in time.
## Use the canvas "File/Quit ROOT" to exit from this example
##
## \macro_code
##
## \author  Olivier Couet
## \translator P. P.


import ROOT
import ctypes

#classes
TCanvas = ROOT.TCanvas
TMath = ROOT.TMath
TGraph = ROOT.TGraph

#standar library
std = ROOT.std
vector = std.vector
rotate = std.rotate
std_next = std.next

#constants
kBlue = ROOT.kBlue
kGreen = ROOT.kGreen
kCanDelete = ROOT.kCanDelete

#types
c_double = ctypes.c_double

#globals
gSystem = ROOT.gSystem
gROOT = ROOT.gROOT

#system utils
Remove = ROOT.gROOT.Remove


#utils
def to_c( ls ):
   return ( c_double* len(ls))( * ls )
def to_py( c_ls):
   return list(c_ls)

# void
def gtime() :

   global c1
   c1 = gROOT.FindObject("c1") # TCanvas
   if (c1) : Remove(c1); del c1
   
   c1 = TCanvas("c1")
   ng = 100
   kNMAX = 10000
   global X, Y
   X = vector["Double_t"](kNMAX)
   Y = vector["Double_t"](kNMAX)

   global cursor
   cursor = kNMAX + 0 # Int_t
   x = 1.; stepx = 0.1 # Double_t
   
   while (not gSystem.ProcessEvents()) :

      if (cursor + ng >= kNMAX):

         cursor = 0
         #for (Int_t i = 0; i < ng; i++) {
         for i in range(0, ng, 1): 
            X[i] = x
            Y[i] = TMath.Sin(x)
            x += stepx
            
      else:

         X[cursor+ng] = x
         Y[cursor+ng] = TMath.Sin(x)
         x += stepx
         cursor += 1
         
      
      global g
      #Not to use: 
      #g = TGraph(ng, X.data()+cursor, Y.data()+cursor)
      #Note:
      #      X and Y don't have a operator+() implemented in PyRoot.
      #Not to use:
      #g = TGraph(ng, X[cursor: ].data(), Y[cursor:].data())
      #Note:
      #      There is aproblem with the cppyy.LowLevelView(). TGraph-class 
      #      confuses X by Y; so when we are plotting, the graph looks like X=X.
      #Instead:
      c_X = to_c( list( X[cursor:ng+cursor] ) )
      c_Y = to_c( list( Y[cursor:ng+cursor] ) )
      g = TGraph(ng, c_X, c_Y )

      g.SetMarkerStyle(21)
      g.SetMarkerColor(kBlue)
      g.SetLineColor(kGreen)
      g.SetBit(kCanDelete); # let canvas delete graph when call TCanvas.Clear()
      
      c1.Clear()
      g.Draw("alp")
      c1.Update()
      #print("cursor ", cursor)
      
      gSystem.Sleep(10)
      
   


if __name__ == "__main__":
   gtime()
