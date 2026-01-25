## \file
## \ingroup tutorial_tree
## \notebook -nodraw
##
##
##  Read data from an ascii file and create a root file with an histogram 
##  and an ntuple.
##  See a variant of this macro in "basic2.py".
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
                   TFile,
                   TNtuple,
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
def basic() :

   """
   Read file $ROOTSYS/tutorials/tree/basic.dat.
   which has 3 columns of float data, like:
   1.123   2.345  3.456
   4.223   5.345  6.456
   7.123   8.345  9.456
   ...     ...    ...

   """

   #
   TutDir = gROOT.GetTutorialDir()  # TString
   Dir = TutDir.Data() 
   Dir = Dir + "/tree/" 
   Dir = Dir.replace("/./", "/")
   #
   file_in = open( "%sbasic.dat" % Dir )
   

   #
   global f, h1, ntuple
   f      = TFile.Open("basic.root", "RECREATE")  # auto
   h1     = TH1F("h1", "x distribution", 100, -4, 4)
   ntuple = TNtuple("ntuple", "data from ascii file", "x:y:z")
   
   #
   nlines = 0  # Int_t
   #
   while (1)  :
      #
      line = file_in.readline()

      #
      if not line : 
         break
      #
      x, y, z = map( float, line.split() )
         
      #
      if ( nlines < 5 )  :
         printf("x=%8f, y=%8f, z=%8f\n",
                 x,
                 y,
                 z,
                 )
         
      #
      h1.Fill(x)
      ntuple.Fill(x, y, z)

      #
      nlines += 1
      
   #
   printf(" Found %d points.\n", nlines)
   
   #
   file_in.close()
   # 
   f.Write()
   f.Close()

   #
   f.Clear()
   f.Delete()
   gROOT.Remove( f )
   del f
   


if __name__ == "__main__":
   basic()
