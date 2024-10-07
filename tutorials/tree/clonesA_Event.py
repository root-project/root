## \file
## \ingroup tutorial_tree
##
##
## Example to write and read a Tree built with a complex class inheritance tree.
## It demonstrates usage of inheritance and TClonesArrays conjointly. 
## This is a simplified-stripped extract of an event structure which was used
## within the Marabou project.
##
## To run this example, do:
## ~~~
##  IPython [1]: %run clonesA_Event.py
## ~~~
##
##
## \macro_code
##
## \author The ROOT Team
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


# seamlessly integration with c
from ROOT.gROOT import ProcessLine




#
# Loading complex classes.
#
global TUsrSevtData1, TUsrSevtData2
#
try :
   TUsrSevtData1 = ROOT.TUsrSevtData1
   TUsrSevtData2 = ROOT.TUsrSevtData2
except:
   #
   s1 = os.path.abspath( __file__ )
   #
   Dir = gSystem.UnixPathName(
            s1[ 0 : s1.rfind("/") ]
         ) # str 

   #
   ProcessLine( ".L " + Dir + "/clonesA_Event.cxx+" )
   #
   TUsrSevtData1 = ROOT.TUsrSevtData1
   TUsrSevtData2 = ROOT.TUsrSevtData2
   #
   ProcessLine("clonesA_Event(true)")
   



# void
def clonesA_Event_w() :

   #
   # Protect against old ROOT versions.
   if (gROOT.GetVersionInt() < 30503)  :
      print( "Works only with ROOT version >= 3.05/03")
      return
      
   if (gROOT.GetVersionDate() < 20030406)  :
      print( "Works only with ROOT CVS version after 5. 4. 2003")
      return
      
   
   #
   # Write a Tree.
   #
   global hfile, tree
   hfile = TFile("clonesA_Event.root", "RECREATE", "Test TClonesArray")  # TFile
   tree  = TTree("clonesA_Event", "An example of a ROOT tree")           # TTree
   #
   event1 = TUsrSevtData1()  # TUsrSevtData1
   event2 = TUsrSevtData2()  # TUsrSevtData2
   #
   tree.Branch("top1", "TUsrSevtData1", event1, 8000, 99)
   tree.Branch("top2", "TUsrSevtData2", event2, 8000, 99)


   #
   # Fill the Tree.
   #
   #for (Int_t ev = 0; ev < 10; ev++) {
   for ev in range(0, 10, 1):
      #
      print( "event :" , ev)
      #
      event1.SetEvent(ev)
      event2.SetEvent(ev)
      #
      tree.Fill()
      #
      if (ev < 3)  :
         tree.Show(ev)
         
      
   #
   tree.Write()
   tree.Print()

   #
   hfile.Close()
   gROOT.Remove( hfile )
   del hfile
   

# void
def clonesA_Event_r() :

   #
   # Read the Tree.
   #
   hfile = TFile("clonesA_Event.root")  # TFile
   tree  = hfile.Get("clonesA_Event")   # (TTree *)
   
   #
   event1 = TUsrSevtData1() #  * # 0 
   event2 = TUsrSevtData2() #  * # 0
   #
   tree.SetBranchAddress("top1", event1)
   tree.SetBranchAddress("top2", event2)

   
   #
   #for (Int_t ev = 0; ev < 8; ev++) {
   for ev in range(0, 8, 1):
      #
      tree.Show(ev)
      #
      print( "Pileup event1: " , event1.GetPileup() )
      print( "Pileup event2: " , event2.GetPileup() )
      event1.Clear()
      event2.Clear()

      # Detect possible memory leaks.
      # gObjectTable.Print()           
      
   #
   gROOT.Remove( hfile )
   del hfile
   

# void
def clonesA_Event() :

   # Write the Tree.
   clonesA_Event_w()  
   
   # Read back the Tree.
   clonesA_Event_r()  
   



if __name__ == "__main__":
   clonesA_Event()
