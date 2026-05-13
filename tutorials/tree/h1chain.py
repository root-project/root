## \file
## \ingroup tutorial_tree
## \notebook -nodraw
##
##
## Creates a TChain object to be used by the "h1analysis.py" script.
## The symbol "H1" (a string) must point to a directory where the "H1" data sets
## have been installed.
##
##
## \macro_code
##
## \author Rene Brun
## \translator P. P.


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
                   gSystem,
                   gStyle,
                   gPad,
                   gRandom,
                   gBenchmark,
                   gROOT,
)


# variables
chain = TChain("h42")


# void
def h1chain(h1dir : Char_t = 0) :

   #
   if (h1dir)  :
      gSystem.Setenv("H1", h1dir)
      

   #
   chain.SetCacheSize(20 * 1024 * 1024)

   #
   chain.Add("$H1/dstarmb.root")
   chain.Add("$H1/dstarp1a.root")
   chain.Add("$H1/dstarp1b.root")
   chain.Add("$H1/dstarp2.root")
   


if __name__ == "__main__":
   h1chain()
