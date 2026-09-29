## \file
## \ingroup tutorial_tree
## \notebook -nodraw
##
##
## Copy a subset of a Tree (one branch) to a new Tree in a separate file.
##
## Description and details:
##    One branch of the new Tree is written into a separate file.
##    The input file 'tree4.root' is being generated, instead of 
##    the old "event.root" file,
##    with the help of another tutorial: 'tree4.py'. 
##    If you want have the "event.root" file as in the .C version,
##    use the "Event.py" script.
##  
##
## \macro_code
##
## \author Rene Brun
## \translator P. P.


import os
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
                   TFile,
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
                   gSystem,
                   gStyle,
                   gPad,
                   gRandom,
                   gBenchmark,
                   gROOT,
)



#void
def copytree2() :


   # 
   # Old ROOT data.
   #
   # Dir = "$ROOTSYS/test/Event.root"  # TString
   # gSystem.ExpandPathName(Dir)
   # filename = "./Event.root" if gSystem.AccessPathName(Dir) else "$ROOTSYS/test/Event.root" # auto
   # 

   # Different Approach. Creating a .root file with "Event" class type.
   #
   print ( " >> >> >> >> >> >> >> >> >> >>  " )
   print ( " >> >> >> >> >> >> >> >> >> >>  " )
   print ( " >> >> >> >> >> >> >> >> >> >>  " )
   print ( " >> >> >> Generating tree4.root file for analysis." )

   from tree4 import tree4w
   tree4w()

   print ( " << << << << << << << << << <<  " )
   print ( " << << << << << << << << << <<  " )
   print ( " << << << << << << << << << <<  " )
   print ( " << << << tree4.root file generated sucessfully." )
   print ( "          Coming back to 'copytre2.py' analysis. " )
   print ( " " )

   #

   Dir = "$ROOTSYS/tutorials/tree/tree4.root"  # TString
   Dir = os.path.expandvars( Dir )
   #
   if os.path.exists( Dir ) :
      filename = Dir
   elif os.path.exists( "./tree4.root" ) :
      filename = "./tree4.root" 
   else :
      raise FileNotFoundError( "tree4.root" )
   # 
   

   #
   oldfile = TFile( filename )
   oldtree = TTree() # nullptr # * 
   # oldfile.GetObject["TTree"]("T", oldtree) # ROOT 5 with Event and "root/test/."
   oldfile.GetObject["TTree"]("t4", oldtree)
   
   # Activate only four of thm.
   for activeBranchName in [ 
                             # "event", # ROOT 5
                             "event_not_split",
                             "fNtrack",
                             "fNseg",
                             "fH",
                             ]:
      oldtree.SetBranchStatus(activeBranchName, 1)
      
   
   #
   # Create a new file + a clone from the old tree header. 
   # Do NOT copy events YET.
   newfile = TFile("small.root", "recreate")
   newtree = oldtree.CloneTree(0)  # auto
   
   #
   # Redirect branch fH into a separate file and copy all its events.
   newtree.GetBranch("fH").SetFile("small_fH.root")
   newtree.CopyEntries(oldtree)
   
   #
   newtree.Print()
   newfile.Write()

   #
   newfile.Close()
   


if __name__ == "__main__":
   copytree2()
