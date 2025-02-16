## \file
## \ingroup tutorial_tree
## \notebook
##
##
## A simple use of the TSpider class. 
##
##
## \macro_code
##
## \author Bastien Dallapiazza
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
                   TFile,
                   TNtuple,
                   TSpider,
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
def spider() :

   #
   global c1, f
   c1 = TCanvas("c1", "TSpider example", 200, 10, 700, 700)  # TCanvas
   f = TFile("hsimple.root")  # TFile

   # Check in advance.
   if (not f or f.IsZombie())  :
      printf("Please, run $ROOTSYS/tutotrials/tree/hsimple.py before.")
      return
      

   # TSpider needs a ntuple.
   global ntuple
   ntuple = f.Get("ntuple")  # (TNtuple *)


   #
   # Arguments for TSpider constructor.
   #
   varexp    = "px:py:pz:random:sin(px):log(px/py):log(pz)"  # TString
   selection = "px>0 && py>0 && pz>0"                            # TString
   options   = "average"                                     # TString


   # 
   # Simple construction.
   #
   global spider
   spider = TSpider(ntuple, varexp, selection, options)  # TSpider
   #
   spider.Draw()

 
   #
   c1.ToggleEditor()
   c1.Selected(c1, spider, 1)
   c1.Update()
   c1.Draw()
   


if __name__ == "__main__":
   spider()
