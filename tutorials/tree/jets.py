## \file
## \ingroup tutorial_tree
##
##
## Usage of a Tree using the JetEvent class.
##
## The JetEvent class has several member collections of the
## type TClonesArray, and also has   
## another member collections of the type TRefArray which 
## reference to objects in those TClonesArrays.
##
## For technical details, you can find the JetEvent class
## in the $ROOTSYS/tutorials/tree/JetEvent.h,cxx source code.
##
## To execute this script, do:
## ~~~
## IPython [1]: %run jets.py # Loads the JetEvent module and runs the script.
## IPython [2]: %run jets.py # Runs the script only. 
## IPython [3]: %run jets.py # Does nothing.
## IPython [4]: %run jets.py # Does nothing.
## ~~~
##
##
## \macro_code
##
## \author Rene Brun
## \translator P. P.


import sys
from functools import singledispatch
import ctypes
from array import array

import ROOT


from ROOT import (
                   TFile,
                   TROOT,
                   TRandom,
                   TSystem,
                   TTree,
                   )

# standard library
from ROOT import std
from ROOT.std import (
                       make_shared,
                       unique_ptr,
                       )

# classes
from ROOT import (
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




#
# Loading JetEvent Class efficiently.
#
ipython = get_ipython()
if hasattr(ipython, 'JETS_SECOND_RUN') and \
               not ipython.JETS_SECOND_RUN :
   #
   print( " Stop loading modules. It is second run.",
          " They already have been loaded. "
          )
   ipython.JETS_SECOND_RUN = True
   JetEvent = ROOT.JetEvent
   #
   # It continues with the script.
   
elif hasattr(ipython, 'JETS_SECOND_RUN') and \
                     ipython.JETS_SECOND_RUN :
   #
   # JETS_SECOND_RUN = True # A counter would be better. 
   print( " Do nothing. More than one run." )
   sys.exit()

else :
   ipython.JETS_SECOND_RUN = False 
   #
   # Loading modules.
   #
   print( " Loading modules JetEvent." )
   tutdir = gROOT.GetTutorialDir().Data()  # TString().Data() # str
   gROOT.ProcessLine(".L " + tutdir + "/tree/JetEvent.cxx+") 
   #
   JetEvent = ROOT.JetEvent
   #
   # It continues with the script.



# void
def write(nev : Int_t = 100) :

   """
   Write new Jet events.

   """

   #
   global f, T
   f = TFile("JetEvent.root", "recreate")
   T = TTree("T", "Event example with Jets")  # TTree

   #
   global event
   event = JetEvent()  # JetEvent
   #
   T.Branch("event", "JetEvent", event, 8000, 2)

   
   # Generate events and fill them on the Tree.
   # 
   #for (Int_t ev = 0; ev < nev; ev++) {
   for ev in range(0, nev, 1):
      event.Build()
      T.Fill()
      
   
   #
   T.Print()
   T.Write()

   #
   f.Close() 
   

# void
def read() :

   """
   Read the JetEvent file and get its information.

   """

   #
   global f, T
   f = TFile("JetEvent.root")
   T = f.Get("T")  # (TTree *)

   #
   event = JetEvent() # * # nullptr
   T.SetBranchAddress("event", event)


   #
   # Getting Information.
   nentries = T.GetEntries()  # Long64_t
   # 
   #for (Long64_t ev = 0; ev < nentries; ev++) {
   for ev in range(0, nentries, 1):
      #
      T.GetEntry(ev)
      #
      if (ev) :
         # dump first event only
         continue  
      #
      print( " Event: "                     ,
                          ev                ,
             " Jets: "                      ,
                          event.GetNjet()   ,
             " Tracks: "                    ,
                          event.GetNtrack() ,
             " Hits A: "                    ,
                          event.GetNhitA()  ,
             " Hits B: "                    ,
                          event.GetNhitB()  ,

             ) 
      
   

# void
def pileup(nev : Int_t = 200) :

   """
   Create pileup events where each build have selected events
   randomly among the nentries with "LOOPMAX" limit.

   """

   #
   global f, T
   f = TFile("JetEvent.root")
   T = f.Get("T")  # (TTree *)


   # nentries = T.GetEntries()  # Long64_t 

   #   
   LOOPMAX = 10 # Int_t
   #
   events = [ JetEvent() for _ in range( LOOPMAX ) ]
   #
   #
   #for (Long64_t ev = 0; ev < nev; ev++) {
   for ev in range(0, nev, 1):
      #
      if (ev % 10 == 0)  :
         printf("Building pileup: %d\n", ev)
      #   
      # for (loop = 0; loop < LOOPMAX; loop++) {
      for loop in range(0, LOOPMAX, 1):
         #
         T.SetBranchAddress("event", events[ loop ])
         #
         rev = Int_t( gRandom.Uniform( LOOPMAX ) ) 
         T.GetEntry(rev)
         
      
   
def jets(nev : Int_t = 100, npileup : Int_t = 200 ) :

   write(nev)
   read()
   pileup(npileup)

   # Note :
   #        Embedding these loads inside the first run of the script,
   #        could be done by using ipython. But it is not the
   #        purpose of the current tutorial.
   



if __name__ == "__main__":
   jets()
