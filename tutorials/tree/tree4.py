## \file
## \ingroup tutorial_tree
## \notebook -nodraw
##
## This example writes a Tree with TEvent objects onto a .root file. 
## It is a simplified version of the $ROOTSYS/test/MainEvent.cxx macro
## to write the Tree, and also uses snippets of code of 
## the $ROOTSYS/test/eventb.C macro to generate events.
## 
## In particular, this example shows:
##   - How to fill a Tree with an event class containing these data members:
##     ~~~
##         char           fType[20];
##         Int_t          fNtrack;
##         Int_t          fNseg;
##         Int_t          fNvertex;
##         UInt_t         fFlag;
##         Float_t        fTemperature;
##         EventHeader    fEvtHdr;
##         TClonesArray  *fTracks;            #->
##         TH1F          *fH;                 #->
##         Int_t          fMeasures[10];
##         Float_t        fMatrix[4][4];
##         Float_t       *fClosestDistance;   #[fNvertex]
##     ~~~
##   - The difference in splitting or not splitting a branch.
##   - How to read certain selected branches of the Tree, and 
##     print the first entry with less than 587 tracks.
##   - How to browse and analyze the Tree via the TBrowser 
##     and TTreeViewer classes.
##
## This example can be run in many different ways:
##  -Way 1: using the Cling interpreter:
##         ~~~
##         .x tree4.C
##         ~~~
##  -Way 2: using the Cling interpreter:
##         ~~~
##         .L tree4.C
##         tree4()
##         ~~~
##  -Way 3: using ACLIC:
##         ~~~
##         .L ../test/libEvent.so
##         .x tree4.C++
##         ~~~
##  -Way 4: 
##         One can also run the write and read parts in two separate sessions.
##         For example following one of the sessions above, one can start the session:
##         ~~~
##         .L tree4.C
##         tree4r();
##         ~~~
##
## \macro_code
##
## \author Rene Brun
## \translator P. P.


import numpy as np
from array import array
import ctypes

import ROOT



# standard library
from ROOT import std
from ROOT.std import (
                       make_shared,
                       unique_ptr,
                       )

# classes
from ROOT import (
                   TBrowser,
                   TClassTable,
                   TFile,
                   TH2,
                   TROOT,
                   TRandom,
                   TSystem,
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
                   UInt_t,
                   nullptr,
                   char,
                   )
from ctypes import (
                     c_double,
                     c_float,
                     create_string_buffer,
)


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



#
# It won't work ...
# gSystem.Load( "libEvent.so" ) 
#
# Inherited from .C version:
# #include "../test/Event.h" 
#
# So, 
# Event =  ... ? # How to get it?
#
# The files Event* to compile libEvent.so are located in 
# root_src/test/Event* 
# Since, you are using pyroot, it is likely that you just 
# installed using "pip" or "brew" or maybe you have 
# downloaded the binary pre-compiled files from 
# https://root.cern/install/all_releases/
# Anyhow, it is likely that you don't have access to 
# Event* files, worse libEvent.so.
# We will trust that you have internet connection and
# we will use the power of scripting language that is 
# python...
# 
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 
print( "Geting and Compiling libEvent.so with ACLiC.")
import subprocess
import os
import shutil

if not os.path.isfile( "./Event/Event_cxx.so" ):
   
   #
   # If you are advanced in ROOT, you probably know this by hand.
   root_url = "https://raw.githubusercontent.com/root-project/root/master/"
   event_files = [ "Event.cxx",
                   "Event.h",
                   "EventLinkDef.h",
                   "EventMT.cxx",
                   "EventMT.h"
                   ]
   
   
   # Event* files will be downloaded at ./Event.
   new_dir_Event = "./Event"
   os.makedirs(new_dir_Event, exist_ok=True)
   
   # Changing to the new ./Event directory.
   os.chdir( new_dir_Event )
   
   
   # And, `curl` and raw-git-hub-user-content will do the rest 
   #       to get the Event* files.
   # 
   for event_file in event_files:
      try:
         subprocess.run( [ "curl",
                           "-O",
                           root_url + "test/" + event_file,
                           ],
         
                         check = True,
         
                         )
         print( event_file, "sucessfully downloaded at ./Event .")
      except:
         Warning( event_file, "hasn't been sucessfully downloaded at Event.")
   print("\n\n >>> Files were downloaded.")
   
   #
   # Moving Event* files to root/include directory. 
   # There was no other option,  AddIncludePaths, AddDynamicPath weren't working well.
   #
   ROOTSYS = str( gROOT.GetDataDir() )
   #
   for event_file in event_files:
      #os.rename( event_file, ROOTSYS + "/include/" + event_file )
      shutil.copy( event_file, ROOTSYS + "/include/" + event_file )
   
   
   print( "\n >>> Compiling Files with Cling.\n\n" )
   #
   # Compiling with Cling.
   #
   # Formating : "{ Event.h, Event.cxx, ... } "
   #source_Event = "{" + ", ".join( event_files ) + "}"
   source_Event = "Event.cxx"
   #
   # Compile the source files into a shared library
   #gSystem.AddDynamicPath( os.getcwd() )
   #gSystem.AddIncludePath( os.getcwd() )
   # No need. It mixes paths.
   
   #gSystem.CompileMacro( source_Event, "gO ")
   gSystem.CompileMacro( source_Event, )
   
   # libEven.so
   #os.rename( "Event_cxx.so", "libEvent.so")
   #os.rename( "Event_cxx.so", ROOTSYS + "lib/libEvent.so")
   #shutil.copy( "Event_cxx.so", ROOTSYS + "lib/libEvent.so")
   #os.rename( "Event_cxx.so", "libEvent.so")
   
   os.chdir("..")
   #
   #gSystem.GetDynamicPath( )
   
   
   #import sys
   #sys.exit()
else :
   print( " Shared library Event already compiled.")
print( "Loading Event library ...") 

   
   
# ...
#
# Now, you can use libEvent.so;
# Not without adding ./Event to the dynamic library search path.
#
gSystem.AddDynamicPath( "./Event" )
# No need. It seems ACLiC already does that.
#
#ROOT.gSystem.Load("libEvent.so") # Inherited from .C Version.
try :
    Event = ROOT.Event
except:
    
    ROOT.gSystem.Load("Event_cxx.so")
    Event = ROOT.Event
print( "... Done.")

#
#

#void
def tree4w() :
   
   # Create a Tree file tree4.root file.
   global f
   f = TFile("tree4.root", "RECREATE")
   
   # Create a ROOT Tree.
   global t4
   t4 = TTree("t4", "A Tree with Events")
   
   # Create a pointer to an Event object.
   global event
   event = Event()  # Event
   
   # Create two branches, split one.
   t4.Branch("event_split"     , event , 16000 , 99 )
   t4.Branch("event_not_split" , event , 16000 , 0  )
   # Note:
   #       - 16000 is size of the buffer; Number of 
   #         Entries to hold before written to the 
   #         file.
   #       - 99 and 0 are the degree of splitting.
   #         0 means no splitting at all.
   #       

   
   # A local variable for the event type.
   etype = [ char() for _ in range(20) ]
   etype = create_string_buffer( 20 ) 
   
   #
   # Fill the tree.
   #
   #for (Int_t ev = 0; ev < 100; ev++) {
   for ev in range(0, 100, 1):
      #
      sigmat = np.float64() # Float_t()
      sigmas = np.float64() # Float_t()
      gRandom.Rannor(sigmat, sigmas)
      #
      random = gRandom.Rndm(1)  # Float_t
      #
      etype = sprintf(etype, "type%d", ev % 5)
      #
      ntrack = Int_t(600 + 600 * sigmat / 120.)  # Int_t
      #
      event.SetTemperature ( random + 20.                       )
      event.SetNvertex     ( Int_t( 1 + 20 * gRandom.Rndm() )   )
      event.SetHeader      ( ev, 200, 960312, random            )
      event.SetType        ( etype                              )
      event.SetNseg        ( Int_t( 10 * ntrack + 20 * sigmas ) )
      event.SetFlag        ( UInt_t( random + 0.5 )             )
      #
      # 
      #      for (UChar_t m = 0; m < 10; m++) {
      for m in range(0, 10, 1):
         event.SetMeasure(m, Int_t(gRandom.Gaus(m, m + 1)))
         
      
      #
      # Fill the matrix.
      #
      #      for (UChar_t i0 = 0; i0 < 4; i0++) {
      for i0 in range(0, 4, 1):
         #         for (UChar_t i1 = 0; i1 < 4; i1++) {
         for i1 in range(0, 4, 1):
            event.SetMatrix(i0, i1, gRandom.Gaus(i0 * i1, 1))
            
         
      
      #
      #  Create and fill the Track objects.
      #
      #      for (Int_t t = 0; t < ntrack; t++) {
      for t in range(0, ntrack, 1):
         event.AddTrack(random)
         
      
      # Fill the tree.
      t4.Fill()
      # Warning : The Tree is filled
      #           Not the branches individually.
      #           Filling the tree will actually 
      #           fill both branches, the 
      #           splitting one and the one 
      #           with normal size(zero splitting).
      #      
      #           If you fill only one branch, 
      #           whichever, it could lead to 
      #           unexpected memory behaviour.
      #           Simple cases like this 
      #           will not raise any problems,
      #           advanced analysis could lead
      #           false results.
      #          
      #           The .Fill method of TBranchElemnt 
      #           and Tree both have the same behaviour
      #           In filling the entire Tree. However,
      #           TBranch.Fill only updates their values.
      #           of the last entry being called.
      #           So, no new entry is created,
      #           and the values of the other branches
      #           remain intact. 
      #
      
      
      # Clear the event before reloading it.
      event.Clear()
      
   

   # Write the file header.
   f.Write()
   
   # Print the tree contents.
   t4.Print()

   # Deleting objects from ROOT 
   # for running this script again.
   gROOT.Remove( event )
   gROOT.Remove( f )
   gROOT.Remove( t4 )
   


# void
def tree4r() :

   """
   Read the tree generated with tree4.
   
   """

   #
   # Note that we use "global" to before creating 
   # the TFile and TTree objects!
   # because we want to keep these objects alive 
   # when we leave this function.
   global f, t4
   f  = TFile("tree4.root") # TFile
   t4 = f.Get("t4")         # (TTree *)
   
   #
   # Create a pointer to an event object. This will be used
   # to read the branch values.
   global event
   # event = Event()  # Event 
   #
   # Error: not to create a new Event. 
   #        Event_cxx.so aka libEvent.so hasn't the 
   #        descructor ~Event implemented.
   # 
   #        Or, you can simply create another event, 
   #        let's say: event_for_tracks:
   #           global event_for_tracks
   #           event_for_tracks = Event()
   

   #
   # The error is not a matter of crossing references. This will do 
   # but the problem persists.
   global event_tracks
   event_tracks = Event()  # Event 
   #
   # This Not solves the issue as mentioned above.
   # bntrack.SetAddress(event_tracks) # Correct the .C version. This line was added here.
   

   #
   # Get the two branches.
   #
   global bntrack, branch
   branch  = t4.GetBranch("event_split" ) # TBranch *
   bntrack = t4.GetBranch("fNtrack"     ) # TBranch *


   #
   # Set the branch address on "bntrack" only. ...
   #
   # branch.SetAddress( event )                   # error
   # branch.SetAddress( ROOT.AddressOf( event ) ) # error  
   bntrack.SetAddress(event_tracks)               # TODO:Correct the .C version.
   # 
   # ... "branch" has the information of "event_split" now.

   #
   # If you want to get the information from 
   # "event_not_split" branch you should do:
   #    branch  = t4.GetBranch("event_not_split" ) # TBranch *
   #    bntrack = t4.GetBranch("fNtrack"     )     # TBranch *
   #    bntrack.SetAddress(event_tracks)           #
   # 
   

   #
   # Getting information from "event_split" branch
   # into "bntrack". 
   #
   global nevent, nbytes, nselected
   nevent    = t4.GetEntries() # Long64_t
   nselected = 0               # Int_t
   nbytes    = 0               # Int_t
   #
   # The Main Event Loop.
   #
   #for (Long64_t i = 0; i < nevent; i++) {
   for i in range(0, nevent, 1):
      #
      # Read branch "event_split".
      #branch.GetEntry(i) # error
      #
      # Read branch "fNtrack" only.
      bntrack.GetEntry(i) #
      #
      # Note :
      #        In the TBrowser, the only branch(or leaf) able to read is "event_split".
      #        Open any other leaf will cause memory issues.
      #        If you want to see any other branch(or leaf, whatever the case 
      #        corresponds, you will have to load it here:
      #         - Get the branch, 
      #         - Set its address using event=Event(), 
      #         - Obtain its information using .GetEntry( i )
      #
      # Note: 
      #       Since event is a complex data structure,
      #       at getting its entry python will raise error
      #       because event isn't a type recognized by
      #       pyhon. Even so, a c-type.
      #


      #
      # Accessing information completed.
      # Now, let's make some analysis.
      

      # Reject events with less than a certain number of tracks.
      #
      #if (event_tracks.GetNtrack() < 587 )  :  # Poorly selection. 
      #if (event_tracks.GetNtrack() < 600 )  :  # ~ 40 selected
      #if (event_tracks.GetNtrack() > 580 )  :  # good selection.
      if (event_tracks.GetNtrack() > 598 )  :  #  good selection.
      #if (event_tracks.GetNtrack() > 605 )  :  #  good selection.
         continue
      # Note :
      #        To choose a threshold, check out the histogram in 
      #        the leave fNtrack using the TBrowser. 
      

      
      #
      # Read the complete accepted event from the memory if
      # the event has been accepted.
      #
      nbytes    += t4.GetEntry(i)
      nselected += 1
      #
      # Note: 
      #      This is not a good analysis.
      #      Better would be:
      #         sel_branches_idxs  = [ ] 
      #         sel_branches_idxs.append( i )
      
      #
      # Print only the first accepted event.
      if (nselected == 1)  :
         t4.Show()
    
      #
         
      
      #
      # Clear tracks' array.
      #
      event_tracks.Clear()
      #event.Clear()        # error
      #
      
   
   #
   if (gROOT.IsBatch())  :
      return
      
   #
   global new_browser
   new_browser = TBrowser()
   t4.StartViewer()
   
   #
   # To avoid memory leaks.
   #
   gROOT.Remove( event_tracks )
   gROOT.Remove( f )
   gROOT.Remove( t4 )
   gROOT.Remove( branch ) 
   gROOT.Remove( bntrack )
   gROOT.Remove( new_browser )
   # 
   # Note:
   #       At rerunning this script there is no problem 
   #       at all, since we delete the ownership from ROOT
   #       of the ROOT objects: f, t4, branch, ... .
   #       However, at exiting ipython, there are still
   #       some memory issues regardis this issue.
   #       FIXME
   #       Mainly is due to the TBrowser object.
   #       we have to close it manually.
   #       
   #       


# void
def tree4() :

   # First write.
   global Event
   Event.Reset()  # Allow for re-run this script by cleaning static variables.
   tree4w()

   # Then read.
   Event.Reset()  # Allow for re-run this script by cleaning static variables.
   tree4r()
   


if __name__ == "__main__":
   tree4()
