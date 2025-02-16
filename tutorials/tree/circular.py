## \file
## \ingroup tutorial_tree
## \notebook -nodraw
##
## Example of a circular Tree.
##
## Description:
##    Circular Trees are interesting in online real time environments
##    to store the results of the last "maxEntries" events,
##    where Circular Trees are memory resident.
##    For more info, see TTree.SetCircular.
## 
##    < https://root.cern/doc/master/classTTree.html#a16b26ce06d95d52d99ad5a1bfcb8f4f0 \>
##
##
## \macro_code
##
## \author Rene Brun
## \translator P. P.


import numpy as np
import ctypes
from array import array

import ROOT


# standard library
from ROOT import std
from ROOT.std import (
                       make_shared,
                       unique_ptr,
                       )

# classes
from ROOT import (
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



# void
def circular() :

   #
   global T
   T =  TTree("T", "test circular buffers"); # auto # new

   #
   global r
   r = TRandom()

   #
   global px, py,pz
   global randomNum
   global i
   #
   px, py, pz  = ( np.zeros( 1, dtype="f" ) for _ in range(3) )  # Float_t()
   #
   randomNum   = np.zeros( 1, dtype="d" )                        # Double_t()
   #
   i           = np.zeros( 1, dtype  = np.uint16 )               # UShort_t()


   #
   # Setting-up branches.
   #
   T.Branch ( "px"     , px        , "px/F"     )
   T.Branch ( "py"     , py        , "px/F"     )
   T.Branch ( "pz"     , pz        , "px/F"     )
   T.Branch ( "random" , randomNum , "random/D" )
   T.Branch ( "i"      , i         , "i/s"      )



   # - - - * - - -
   # Setting Circular definition. 
   #
   T.SetCircular(20000)  # keep a maximum of 20000 entries in memory



   # 
   #for (i = 0; i < 65000; i++) { # Rannor  Gaus
   for i in range(0, 65000, 1):  # error # ok 
   #for i in range(0, 650, 1):    # ok    # ok
   #for i in range(0, 1650, 1):   # ok   # ok 
      #
      #r.Rannor(px, py) # error in loop over ~10000
      ##
      px[0] = r.Gaus(0,1)
      py[0] = r.Gaus(0,1)

      #pz[0]      = ( px * px + py * py )[0]
      pz[0]       = px[0] **2  +  py[0] **2

      randomNum[0] = r.Rndm()

      ##
      T.Fill()
      
   T.Print()
   


if __name__ == "__main__":
   circular()
