## \file
## \ingroup tutorial_tree
## \notebook
##
##
## Illustrates how to retrieve TTree variables in arrays (cppyy.LowLevelView).
##
## This example:
##   - Creates a simple TTree.
##   - Generates TTree variables thanks to the `.Draw` method with `goff` option.
##   - Retrieves some of them in arrays thanks to `GetVal` method.
##   - Generates and draws graphs with these arrays.
##
## The option `goff` in `TTree.Draw` behaves like any other drawing option except
## that, at the end, no graphics is produced ( `goff` a.k.a. graphics off). This allows
## This allows to generate as many TTree variables as needed fast without the 
## interruption of graphics. All the graphics options
## (except `para` and `candle`) are limited to four variables only. And `para`
## and `candle` need at least two variables.
##
## Note that by default TTree.Draw creates arrays internally, which can obtained
## with the `.GetVal` method. These arrays have a length corresponding to the parameter 
## `.fEstimate`. By default its value is `fEstimate=1000000` and it can be modified
## via the `TTree.SetEstimate` method. 
## 
## To keep in memory all these results (the ones produced by `fEstimate` and `GetVal`)
## use:
## ~~~{.py}
##   tree.SetEstimate(-1)
## ~~~
##
## Also, `SetEstimate` should be called first if the expected number of selected rows
## surpasses 1000000, the default value.
##
##
## \macro_image
## \macro_output
## \macro_code
##
## \author Olivier Couet
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
from ctypes import ( 
                     c_int,
                     c_double,
                     )

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
def treegetval() :

   """
   Create a simple TTree with 5 branches.

   """


   #
   run, evt = ( np.zeros( 1, "i" ) for _ in range(2) ) # Int_t()
   x, y, z  = ( np.zeros( 1, "f" ) for _ in range(3) ) # Float_t()


   #
   global T
   T = TTree("T", "test friend trees")  # TTree
   #
   T.Branch("Run", run, "Run/I")
   T.Branch("Event", evt, "Event/I")
   T.Branch("x", x, "x/F")
   T.Branch("y", y, "y/F")
   T.Branch("z", z, "z/F")

   #
   global r
   r = TRandom()

   #
   #for (Int_t i = 0; i < 10000; i++) {
   for i in range(0, 10000, 1):
      #
      if (i < 5000)  :
         run[0] = 1
         
      else  :
         run[0] = 2
         
      #
      evt[0] = i
      x[0]   = r.Gaus(10, 1)
      y[0]   = r.Gaus(20, 2)
      z[0]   = r.Landau(2, 1)

      #
      T.Fill()
      
   

   #
   # Draw with option "goff" and generate seven variables.
   # "goff" stands for "Graphics Off".
   n = T.Draw("x:y:z:Run:Event:sin(x):cos(x)", "Run==1", "goff")  # Int_t
   printf("The arrays' dimension is %d\n", n)
   #
   
   # Retrieve variables 0, 5 and 6.
   global vx, vxs, vxc
   vx   = T.GetVal(0) # Double_t * # cppyy.LowLevelView
   vxs  = T.GetVal(5) # Double_t * # cppyy.LowLevelView
   vxc  = T.GetVal(6) # Double_t * # cppyy.LowLevelView
   
   # Create and draw graphs.
   global gs, gc
   gs = TGraph(n, vx, vxs)  # TGraph
   gc = TGraph(n, vx, vxc)  # TGraph
   #
   gs.Draw("ap")
   gc.Draw("p")
   


if __name__ == "__main__":
   treegetval()
