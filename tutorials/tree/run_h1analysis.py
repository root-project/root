## \file
## \ingroup tutorial_tree
## \notebook -nodraw
##
##
## The script driving the analysis can specify the file name and the type.
##
##  - type == 0 : Normal.
##  - type == 1 : Use AClic to compile selector.
##  - type == 2 : Use a fill list and then process the fill list.
##
##
## \macro_code
##
## \author The ROOT Team
## \translator P. P.


import os
import sys
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
                   TChain,
                   TString,
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
                   Char_t,
                   nullptr,
)
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
                   gSystem,
                   gStyle,
                   gPad,
                   gRandom,
                   gBenchmark,
                   gROOT,
)


# For ROOT <= 5.
# if gSystem.Load( "libDavix.so" ) == -1:
#    print( " This script needs libDavix.so library")
#    sys.exit()



#----------------------------------------

# void
def run_h1analysis(type_ : int = 0, h1dir : Char_t = 0) :
   
   #
   print( ">>> Run h1 analysis.")
   
   # First create the chain with all the files.
   chain = TChain("h42")
   
   if (h1dir)  :
      gSystem.Setenv("H1", h1dir)
   else :
      gSystem.Setenv("H1", "http://root.cern.ch/files/h1/")
   


   #
   print( ">>> Creating the chain.")
   
   #
   chain.SetCacheSize(20 * 1024 * 1024)
   #
   chain.Add("$H1/dstarmb.root")
   chain.Add("$H1/dstarp1a.root")
   chain.Add("$H1/dstarp1b.root")
   chain.Add("$H1/dstarp2.root")
   
   #
   # This python script and h1analysis.C macro ...
   # ... in the same directory.
   #
   # selectionMacro = gSystem.GetDirName(__file__).Data() + "/h1analysis.C" # str
   #
   # ... in different directories.
   tut_dir = gROOT.GetTutorialDir().Data() 
   selectionMacro = tut_dir + "/tree/h1analysis.C" # str


   
   #
   if (type_ == 0)  :
      chain.Process( selectionMacro )
   #
   elif (type_ == 1)  :
      #
      # Use AClic by adding "+" character at the end.
      selectionMacro += "+"
      #
      chain.Process( selectionMacro )
   #
   elif (type_ == 2)  :
      chain.Process( selectionMacro , "fillList")
      chain.Process( selectionMacro , "useList")
   else :
      pass
      
   


if __name__ == "__main__":
   run_h1analysis()
