## \file
## \ingroup tutorial_tree
## \notebook -js
##
##
## Create an ntuple filled with data from an ascii file.
## Or how to read data from ascii file and load it into a ntuple.
##
## This script is a variant of basic.py
##
##
## \macro_image
## \macro_code
##
## \author Rene Brun
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
def basic2() :
   #
   TutDir = gROOT.GetTutorialDir()  # TString
   Dir = TutDir.Data() + "/tree/"   # str
   Dir = Dir.replace("/./", "/")    # str
   
   #
   global f, h1, T
   f  = TFile("basic2.root", "RECREATE")           # TFile
   h1 = TH1F("h1", "x distribution", 100, -4, 4)   # TH1F
   T  = TTree("ntuple", "data from ascii file")    # TTree

   #
   nlines = T.ReadFile( "%sbasic.dat" % Dir , "x:y:z")  # Long64_t

   #
   printf(" Found %d points.\n", nlines)

   #
   T.Draw("x", "z>2")

   #
   T.Write()
   f.Close()

   


if __name__ == "__main__":
   basic2()
