## \file
## \ingroup tutorial_tree
## \notebook -nodraw
##
## 
## Get and load "Event" class.
## In case ROOT.Event isn't available or
## ROOT hasn't been built with "libEvent.so",
## which is the case if you obtained a binary version of ROOT.
## ROOT builds the "libEvent.so" in tests directory 
## if you have built it with --test=ON option.
##
## 
##
## \macro_code
##
## \author The ROOT Team
## \author P. P.


import numpy as np
from array import array
import ctypes

import ROOT



from ROOT import (
                  TBrowser, 
                  TClassTable, 
                  TFile, 
                  TH2, 
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
                   UInt_t,
                   nullptr,
                   char,
                   )

# ctypes
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


