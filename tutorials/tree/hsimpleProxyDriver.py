## \file
## \ingroup tutorial_tree
## \notebook -nodraw
##
##
## This is the driver of the "hsimpleProxy" example.
## It provides the infrastructure to run that code on an ntuple
## to be run from the tutorials directory.
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
#
                   kError,
                   kFatal,
)

# globals
from ROOT import (
                   gErrorIgnoreLevel,
                   gSystem,
                   gStyle,
                   gPad,
                   gRandom,
                   gBenchmark,
                   gROOT,
)



# void
def hsimpleProxyDriver() :

   print( "We are at: ", gSystem.WorkingDirectory() )

   #
   global file
   file = TFile.Open("hsimple.root") # TFile *
   #
   if ( not file ) :
      raise RuntimeError( "Input file not found.\n" , "hsimple.root\n", )
   

   #
   global ntuple
   ntuple = TTree() # nullptr # TTree *
   file.GetObject["TTree"]("ntuple", ntuple)

   #
   global Dir
   Dir = gSystem.UnixPathName(
                               __file__[ 0 : __file__.rfind( "/" ) ]

                               )  # str 
   #
   # Same output as  hsimpleProxyDriver.C wiht hsimpleProxy.C .
   #
   # ntuple.Draw( Dir + "/hsimpleProxy.C+")
   #
   # Note :
   #        We are using te .C+ suffix here.
   #        There is not equivalent suffix ".py" 
   #        for the python side.
   #        ntuple.Draw( Dir + "/hsimpleProxy.py")
   #
   #        However, we can achieve similar results
   #        if we use a TTree in between.
   #
   #
   local_scope = { }
   exec( open(Dir + "/hsimpleProxy.py").read(), local_scope )
   #
   # Loading function into the global scope.
   globals()[ "hsimpleProxy" ] = local_scope[ "hsimpleProxy" ]
   #
   # `hsimpleProxy( ntuple )` is now available.
   #  which returns ntuple.px value.
   #
   # Not crete root file for this case.
   # tmp_file = TFile.Open("tmp_file.root", "RECREATE")
   global new_tree
   new_tree = TTree("new_tree", "New Tree with Calculated Values")
   # Since we are only drawing but writing, we could create 
   # a new_tree without creating a file before ( which is a ROOT golden rule).
   # And also, to supress the warnings this carries we shut it down.
   # Or you could just use a temporary TFile.
   new_tree.SetDirectory( 0 )
   #gErrorIgnoreLevel = kFatal 
   gErrorIgnoreLevel = kError 
   #
   # Either: 
   value    = np.zeros( 1, dtype="d" ) 
   new_tree.Branch("calculated_value", value , "value/D")
   # Or:
   # value = std.vector['double'](1)   
   # new_tree.Branch("calculated_value", value)
   #
   for i in range(ntuple.GetEntries()):
      #
      ntuple.GetEntry(i)
      #
      # Call the proxy while loading its value. 
      # Either:
      value[0] = hsimpleProxy( ntuple )  
      # Or:
      # value.clear()
      # value.push_back( hsimpleProxy( ntuple ) )
      #
      #
      new_tree.Fill()
   #
   # Now, instead of :
   # ntuple.Draw( Dir + "/hsimpleProxy.py") # Not available.
   # We will simply Draw the Tree. 
   new_tree.Draw("calculated_value")
   #
   # TODO: Automatize this process in a python function. 
   #       So, we can use it as: 
   #       `ntuple.Draw( Dir + "/hsimpleProxy.py")`



if __name__ == "__main__":
   #
   hsimpleProxyDriver()
