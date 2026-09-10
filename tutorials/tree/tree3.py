## \file
## \ingroup tutorial_tree
## \notebook
##
## An example of how to create a Tree where branches have 
## variable length arrays
## Additionally, how to create and fill a second Tree in parallel.
##
## Run this script with:
## ~~~
##   IPython [1]: %run tree3.py
## ~~~
##
## Little description.
## In the function "treer", the first Tree is open,
## whereas the second Tree is declared as a friend of the first one.
## And the `TTree.Draw` method is called with variables from both Trees.
##
##
## \macro_code
##
## \author Rene Brun
## \translator P. P.


import ctypes
from array import array
import numpy as np

import ROOT



# standard library
from ROOT import std
from ROOT.std import (
                       make_shared,
                       unique_ptr,
                       )

# classes
from ROOT import (
                   TFile,
                   TRandom,
                   TTree,
                   TMath,
                   TCanvas,
                   TPad,
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

# utils
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




# void
def tree3w() :
   
   #
   kMaxTrack  = 500 # Int_t
   #
   ntrack     = np.zeros( 1, dtype="i" ) # Int_t()
   stat       = np.zeros( kMaxTrack, dtype="i" ) # Int_t   [ kMaxTrack ]   
   sign       = np.zeros( kMaxTrack, dtype="i" ) # Int_t   [ kMaxTrack ]   
   px         = np.zeros( kMaxTrack, dtype="f" ) # Float_t [ kMaxTrack ] 
   py         = np.zeros( kMaxTrack, dtype="f" ) # Float_t [ kMaxTrack ] 
   pz         = np.zeros( kMaxTrack, dtype="f" ) # Float_t [ kMaxTrack ] 
   pt         = np.zeros( kMaxTrack, dtype="f" ) # Float_t [ kMaxTrack ] 
   zv         = np.zeros( kMaxTrack, dtype="f" ) # Float_t [ kMaxTrack ] 
   chi2       = np.zeros( kMaxTrack, dtype="f" ) # Float_t [ kMaxTrack ] 
   sumstat    = np.zeros( 1, dtype="d" ) # Double_t()
   
   #
   global f
   f = TFile("tree3.root", "recreate")

   #
   global t3
   t3 = TTree("t3", "Reconst ntuple")  # TTree
   #
   t3.Branch ( "ntrack" , ntrack , "ntrack/I"       )
   t3.Branch ( "stat"   , stat   , "stat[ntrack]/I" )
   t3.Branch ( "sign"   , sign   , "sign[ntrack]/I" )
   t3.Branch ( "px"     , px     , "px[ntrack]/F"   )
   t3.Branch ( "py"     , py     , "py[ntrack]/F"   )
   t3.Branch ( "pz"     , pz     , "pz[ntrack]/F"   )
   t3.Branch ( "zv"     , zv     , "zv[ntrack]/F"   )
   t3.Branch ( "chi2"   , chi2   , "chi2[ntrack]/F" )
   

   #
   global fr
   fr = TFile("tree3f.root", "recreate")

   #
   global t3f
   t3f = TTree("t3f", "a friend Tree")  # TTree
   #
   t3f.Branch ( "ntrack"  , ntrack  , "ntrack/I"     )
   t3f.Branch ( "sumstat" , sumstat , "sumstat/D"    )
   t3f.Branch ( "pt"      , pt      , "pt[ntrack]/F" )
   

   #
   #for (Int_t i = 0; i < 1000; i++) {
   for i in range(0, 1000, 1):
      #
      nt = Int_t( gRandom.Rndm() * (kMaxTrack - 1) )  # Int_t
      #
      ntrack[0]   = nt
      sumstat[0]  = 0
      #
      #for (Int_t n = 0; n < nt; n++) {
      for n in range(0, nt, 1):
         #
         stat[n]  = n % 3
         sign[n]  = i % 2
         #
         px[n]    = gRandom.Gaus(0, 1)
         py[n]    = gRandom.Gaus(0, 2)
         pz[n]    = gRandom.Gaus(10, 5)
         zv[n]    = gRandom.Gaus(100, 2)
         chi2[n]  = gRandom.Gaus(0, .01)
         #
         sumstat[0] += chi2[n]
         #
         pt[n]    = TMath.Sqrt(px[n] * px[n] + py[n] * py[n])
         

      # Fill original Tree.
      t3.Fill()

      # Fill parallel Tree 
      t3f.Fill()
      

   # Principal Tree
   t3.Print()

   #
   # Write first Tree.
   f.cd()
   t3.Write()

   #
   # Write friend Tree.
   fr.cd()
   t3f.Write()

   #
   f.Close()   
   fr.Close()   



# void
def tree3r() :
   # 
   global f, t3
   f = TFile("tree3.root")  # TFile
   t3 = f.Get("t3")  # (TTree *)
   #
   t3.AddFriend("t3f", "tree3f.root")
   t3.Draw("pz", "pt>3")
   

# void
def tree3r2() :
   #
   global p
   p = TPad("p", "p", 0.6, 0.4, 0.98, 0.8)  # TPad
   p.Draw()
   p.cd()

   #
   global f1, f2
   f1 = TFile("tree3.root")  # TFile
   f2 = TFile("tree3f.root")  # TFile

   #
   global t3
   t3 = f1.Get("t3")  # (TTree *)
   #
   t3.AddFriend("t3f", f2)
   #
   t3.Draw("pz", "pt>3")
   

# void
def tree3() :
   tree3w()
   tree3r()
   tree3r2()
   


if __name__ == "__main__":
   tree3()
