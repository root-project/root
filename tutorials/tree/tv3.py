## \file
## \ingroup tutorial_tree
## \notebook
##
##
## How to create and manipulate ROOT trees using a custom class in C++ notation. 
## Specifically, we’ll define a simple Vector3 class to represent a three-dimensional 
## vector position or whatever three coordinates. And use it as base type to create
## and read a tree.
## 
##
## \macro_image
## \macro_code
##
## \author The ROOT Team
## \translator P. P.



import ROOT
import ctypes
from array import array


# standard library
from ROOT import std
from ROOT.std import (
                       make_shared,
                       unique_ptr,
                       )

# classes
from ROOT import (
                   TFile,
                   TTree,
                   TRandom,
                   TCanvas,
                   TH1F,
                   TGraph,
                   TLatex,
)

# maths
from ROOT import (
                   sin,
                   cos,
                   sqrt,
)

# types
from ROOT import (
                   Double_t,
                   Bool_t,
                   Float_t,
                   Int_t,
                   nullptr,
)
#
from ctypes import c_double

#utils
def to_c( ls ):
   return (c_double * len(ls) )( * ls )
def printf(string, *args):
   print( string % args, end="")
def sprintf(buffer, string, *args):
   buffer = string % args 
   return buffer

# constants
from ROOT import (
                   kBlue,
                   kRed,
                   kGreen,
)

# globals
from ROOT import (
                   gStyle,
                   gPad,
                   gRandom,
                   gBenchmark,
                   gROOT,
)



# seamlessly C integration
from ROOT.gROOT import ProcessLine


class Vector3 :

   fX = Double_t()
   fY = Double_t()
   fZ = Double_t()
   
   #public:
   #Vector3() : fX(0), fY(0), fZ(0) {}
   def __init__( self, ):
      pass
   def __init__( self, fX, fY, fZ):
      self.fX = fX 
      self.fY = fY 
      self.fZ = fZ 
   

   # Get... 
   #
   #Double_t
   def x( self, ) :
      return fX;
      
   #Double_t
   def y( self, ) :
      return fY;
      
   #Double_t
   def z( self, ) :
      return fZ;
      
   
   # Set ...
   #
   #void
   def SetXYZ( self, x : Double_t, y : Double_t, z : Double_t) :
      self.fX = x
      self.fY = y
      self.fZ = z
#
# Note : 
#        The above class is a draft of how should be implemented to load
#        it into a Branch. However, the it is still in progress these
#        kind of binding, since it is not trivial.
#        For now we will have to use C-syntax. Or in other words.
#        Define the class in C and pass it over to python using
#        `ProcessLine`.
#
ProcessLine("""
class Vector3 {
   Double_t fX;
   Double_t fY;
   Double_t fZ;

 public:
   Vector3() : fX(0), fY(0), fZ(0) {}

   Double_t x() {
      return fX;
   }
   Double_t y() {
      return fY;
   }
   Double_t z() {
      return fZ;
   }

   void SetXYZ(Double_t x, Double_t y, Double_t z) {
      fX = x;
      fY = y;
      fZ = z;
   }
};
            
""")
Vector3 = ROOT.Vector3
   



# void
def tv3Write() :

   """
   Creates the Tree.

   """

   #
   global v
   v = Vector3()  # Vector3

   #
   global f, T
   f = TFile("v3.root", "recreate")  # TFile
   T = TTree("T", "v3 Tree")  # TTree
   #
   T.Branch("v3", v, 32000, 1)

   #
   global r
   r = TRandom()
   #
   #for (Int_t i = 0; i < 10000; i++) {
   for i in range(0, 10000, 1):
      #
      v.SetXYZ(r.Gaus(0, 1), r.Landau(0, 1), r.Gaus(100, 10))
      #
      T.Fill()
      
   #
   T.Write()
   T.Print()

   #
   f.Close()

   ##
   #gROOT.Remove( f )
   #del f
   


# void
def tv3Read1() :

   """
   First read example showing how to read all branches.

   """

   #
   global v
   v = Vector3() # * # nullptr 

   #
   global f, T
   f = TFile("v3.root")  # TFile
   T = f.Get("T")  # (TTree *)
   T.SetBranchAddress("v3", v)

   #
   global h1, h1_dummy
   h1 = TH1F("x", "x component of Vector3", 100, -3, 3)  # TH1F
   h1_dummy = TH1F("x", "h1_dummy", 100, -3, 3)  # TH1F
   # No run more than 3 times. Error.


   #
   nentries = T.GetEntries()  # Long64_t
   #
   #for (Long64_t i = 0; i < nentries; i++) {
   for i in range(0, nentries, 1):
      #
      T.GetEntry( i )
      #
      h1.Fill( v.x() )
      

   #
   h1.Draw()
   return h1

   

# void
def tv3Read2() :

   """
   Second read example illustrating how to read one branch only.

   """

   #
   global v
   v = Vector3() # * # nullptr

   #
   global f, T
   f = TFile("v3.root")  # TFile
   T = f.Get("T")  # (TTree *)
   #
   T.SetBranchAddress("v3", v)
   #
   global by
   by = T.GetBranch("fY") # TBranch *


   #
   global h2
   h2 = TH1F("y", "y component of Vector3", 100, -5, 20)  # TH1F
   #
   nentries = T.GetEntries()  # Long64_t
   #
   #for (Long64_t i = 0; i < nentries; i++) {
   for i in range(0, nentries, 1):
      by.GetEntry( i )
      h2.Fill( v.y() )
      
   h2.Draw()
   

# void
def tv3() :
   #
   tv3Write()

   #
   global c1
   c1 = TCanvas("c1", "demo of Trees", 10, 10, 600, 800)  # TCanvas
   c1.Divide(1, 2)

   #
   c1.cd(1)
   tv3Read1()
   c1.Update()
   c1.Draw()
   #
   c1.cd(2)
   tv3Read2()
   c1.Update()
   c1.Draw()
   


if __name__ == "__main__":
   tv3()
