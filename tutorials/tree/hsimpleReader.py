## \file
## \ingroup tutorial_tree
## \notebook
##
##
## How to use the TTreeReader class: A simple example.
##
## Description:
## Read data from "hsimple.root" (which was written by "hsimple.py").
##
## \macro_code
##
## \author Anders Eie, 2013
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
                   TH1F,
                   TTreeReader,
                   TTreeReaderValue,
                   TCanvas,
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
def hsimpleReader() :

   # Create a histogram for the values we read.
   global myHist
   myHist =  TH1F("h1", "ntuple", 100, -4, 4)  # auto # new
   
   # Open the file containing the tree.
   global myFile
   myFile = TFile.Open("hsimple.root")  # auto
   if (not myFile or myFile.IsZombie())  :
      return
      
   # Create a TTreeReader for the tree, for instance by passing the
   # TTree's name and the TDirectory / TFile it is in.
   global myReader
   myReader = TTreeReader("ntuple", myFile)
   
   # The branch "px" contains floats; access them as myPx.
   global myPx
   myPx = TTreeReaderValue["Float_t"](myReader, "px")
   # The branch "py" contains floats, too; access those as myPy.
   global myPy
   myPy = TTreeReaderValue["Float_t"](myReader, "py")
   
   # Loop over all entries of the TTree or TChain.
   while (myReader.Next())  :
      # Just access the data as if myPx and myPy were iterators 
      # ( note the use of .Get()[0] ) :
      myHist.Fill(myPx.Get()[0] + myPy.Get()[0] )
      
   
   myHist.Draw()
   


if __name__ == "__main__":
   hsimpleReader()
