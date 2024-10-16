## \file
## \ingroup tutorial_tree
## \notebook -nodraw
##
##
## Create a plot from the data stored in`cernstaff.root`.
## To create `cernstaff.root`, execute tutorial 
## `$ROOTSYS/tutorials/tree/cernbuild.py`
##
## \macro_image
## \macro_code
##
## \author Rene Brun
## \translator P. P.


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
                   TFile,
                   TTree,
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


def is_library_installed( lib_name ):
    try:
        # Attempt to load the library
        ctypes.CDLL(lib_name)
        return True
    except OSError:
        return False

# Check for libgl2ps.so.1.4
if is_library_installed('libgl2ps.so.1.4'):
    print("libgl2ps.so.1.4 is installed.")
else:
    print("libgl2ps.so.1.4 is not installed.")
    sys.exit() 




# void
def staff() :

   #
   global f, T
   f = TFile.Open("cernstaff.root"); # auto
   T = TTree() # TTree *

   #
   # Error:
   # f.GetObject("T", T)
   # Inherited from .C version.
   #    f->GetObject("T", T);
   #    This form doesn't not infer type template.
   # Instead:
   f.GetObject["TTree"]("T", T)
   

   #
   # # Setting up ranges. 
   # minGrade = 0 # Set appropriate min/max values
   # maxGrade = 100 # Set appropriate min/max values
   # minAge = 0 # Set appropriate min/max values
   # maxAge = 100 # Set appropriate min/max values
   # minCost = 0 # Set appropriate min/max values
   # maxCost = 10000 # Set appropriate min/max values
   # #
   # # Add similar for Division and Nation if they are numeric
   # # Set the ranges for the axes
   # T.SetAxisRange(minGrade, maxGrade, "Grade")
   # T.SetAxisRange(minAge, maxAge, "Age")
   # T.SetAxisRange(minCost, maxCost, "Cost")
   # # Add similar for Division and Nation if they a


   T.Draw("Grade:Age:Cost:Division:Nation", "", "gl5d")
   #
   # If you retrieve this error means you need to install "libgl2ps.so.1.4"  .
   # Error :
   # T.Draw("Grade:Age:Cost:Division:Nation", "", "gl5d")
   # In root 6.32.06 :
   #   |  ROOT Version: 6.32.06
   #   |  Built for linuxx8664gcc on Sep 21 2024, 19:19:59
   #   |  From tags/v6-32-06@v6-32-06
   # the library libRGL.so can't found by libRGL.so.
   #
   # Otherwise use the next alternatives.
   # Alternatives:
   # T.Draw("Grade:Age:Cost:Division:Nation", "", "para")
   # T.Draw("Grade:Age:Cost:Division:Nation", "", "candle")



   if (gPad)  :
      gPad.Print("staff_py.png")
   
   #
   f.Close()
   


if __name__ == "__main__":
   staff()
