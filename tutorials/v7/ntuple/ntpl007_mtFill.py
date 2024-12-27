## \file
## \ingroup tutorial_ntuple
## \notebook
## 
## Multi Threaded Fill:
##       Example of multi-threaded writes( multiple threads simultaneously writing 
##       data to a shared resource, such as a file, database, or memory location ) 
##       by using multiple "REntry" objects.
##
## \macro_image
## \macro_code
##
## \date January 2025
## \author The ROOT Team
## \translator P. P.


import concurrent.futures
import threading
import multiprocessing

import numpy as np
import ctypes
import cppyy
import ROOT


# standard library
from ROOT import std
from ROOT.std import (
                       make_shared,
                       unique_ptr,
                       )

# NOTE: The RNTuple classes are experimental at this point.
#       Functionality, interface, and data format is still subject to changes.
#       Do not use for real data!

# root 7
# experimental classes
from ROOT import Experimental
from ROOT.Experimental import (
                                REntry,
                                RNTuple,
                                RNTupleModel,
                                RNTupleReader,
                                RNTupleWriter,
                                )


# classes
from ROOT import (
                   TCanvas,
                   TH1F,
                   TH2F,
                   TRandom,
                   TRandom3,
                   TStyle,
                   TSystem,
                   )

# globals
from ROOT import (
                   gStyle,
                   gPad,
                   gRandom,
                   gBenchmark,
                   gROOT,
                   gInterpreter,
                   )


# Adding colors for the status report output.
# Define color codes.
_RED = "\033[31m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RESET = "\033[0m"


print( _YELLOW + " >>> Loading Wrappers" +
       _RESET )
# C++ wrappers.
#
# Some ROOT 7 classes and their methods don't work
# appropriately in PyRoot.
# We fix this issue by using the cppyy wrapper method.
import subprocess
command = ( " g++                              "
            "     -shared                      "
            "     -fPIC                        "
            "     -o ntpl007_mtFill_wrapper.so      "
            "     ntpl007_mtFill_wrapper.cpp        "
            "     `root-config --cflags --libs`"
            "     -fmax-errors=1"               ,
            )
# Then execute.
try :
   subprocess.run( command, shell=True )
except :
   raise RuntimeError( "ntpl007_mtFill_wrapper.cpp function not well compiled.")

# Then load `ntpl007_mtFill_wrapper.cpp` .
cppyy.load_library( "./ntpl007_mtFill_wrapper.so" )
cppyy.include     ( "./ntpl007_mtFill_wrapper.cpp" )
from cppyy.gbl import (
                        RNTupleWriter_Recreate_Py, 
                        # REntry_Get,    # ... into two specific functions:
                        REntry_Get_vector_float_Py, # one;
                        REntry_Get_uint32_Py,       # and two.

                        std_thread_Py,
                        std_vector_thread_push_back_Py,

                        RNTupleWriter_delete_Py,

                        f_Py, # TODO: Review function signature 
                              #       arguments for extern "C" wrappers.
                              #       Why ?
                        )
# At last, the wrapper functions is ready.
print()


# Where to store the ntuple of this example.
kNTupleFileName = "ntpl007_mtFill.root"

# Number of parallel threads to fill the ntuple.
kNWriterThreads = 4

# Number of events to generate is 
# equal to "kNEventsPerThread" times "kNWriterThreads".
# kNEventsPerThread = 25000 # error
# kNEventsPerThread = 2500 # erro
# kNEventsPerThread = 250 # error
kNEventsPerThread = 25 # error 
# kNEventsPerThread = 11  # ok.
# kNEventsPerThread = 2  # ok.



# Thread function to generate and write events.
# First, setting up global variables.
gLock = std.mutex() 
gThreadId = std.atomic[ std.uint32_t ]() 
#    Note:
#           "gThreadId" could be easily replaced with a simple python "int".
#           Since this script is a translation of ntpl007_mtFill.C, we leave
#           it with std.atomic.
# Then, defining the writer function.
#
# void
def FillData(entry : std.unique_ptr[REntry], ntuple : RNTupleWriter ) :

   global gLock
   global gThreadId

   gThreadId.fetch_add(1)
   threadId = gThreadId.load() # int

   prng = std.make_unique[ TRandom3 ]() # auto
   prng.SetSeed()

   # Not to use: 
   # Id  = entry.Get[ std.uint32_t        ]( "id"  ) # auto
   # vpx = entry.Get[ std.vector[ float ] ]( "vpx" ) # auto
   # vpy = entry.Get[ std.vector[ float ] ]( "vpy" ) # auto
   # vpz = entry.Get[ std.vector[ float ] ]( "vpz" ) # auto
   # Instead:
   # Using the cppyy wrapper functions instead.
   Id  = REntry_Get_uint32_Py(       entry, "id"  )
   vpx = REntry_Get_vector_float_Py( entry, "vpx" )
   vpy = REntry_Get_vector_float_Py( entry, "vpy" )
   vpz = REntry_Get_vector_float_Py( entry, "vpz" )

   
   for i in range(0, kNEventsPerThread, 1):

      vpx.clear()
      vpy.clear()
      vpz.clear()

      Id[0] = threadId  # int
      # Check: Good. Unorderly number of Ids.
      # print( "Id[0]", Id[0] ) 
      
      npx = int( prng.Rndm(1) * 15 )

      # Set the field data for the current event.
      for j in range(0, npx, 1):
         
         px = np.zeros(1, dtype="float")
         py = np.zeros(1, dtype="float")
         prng.Rannor(px, py)
         pz = px * px  +  py * py
         
         vpx.emplace_back[ float ]( px )
         vpy.emplace_back[ float ]( py )
         vpz.emplace_back[ float ]( pz )
         
      # Inherited from C++ version.
      # std::lock_guard<std::mutex> guard(gLock);
      # ntuple->Fill(*entry);

      # TODO: Unnecessary std.lock_guard in Python. Why?
      # guard = std.lock_guard[ std.mutex ] ( gLock )

      # Save.
      # Protect the "ntuple.Fill()" call with "gLock".
      gLock.lock()
      ntuple.Fill( entry )
      gLock.unlock()
   

   # Clean up. Necessary to avoid memory leaks.
   prng.__destruct__()
   
   

# Now, define a function "Write()" to...
# Generate kNEvents with multiple threads in kNTupleFileName.

# void
def Write() :

   # Create the data model.
   global model
   model = RNTupleModel.Create() # auto
   # Field for signature.
   model.MakeField[ std.uint32_t ]( "id" )
   # Fields for momentum $$\vect{p}$$.
   model.MakeField[ std.vector[ float ] ] ( "vpx" )
   model.MakeField[ std.vector[ float ] ] ( "vpy" )
   model.MakeField[ std.vector[ float ] ] ( "vpz" )
   
   # We hand-over the data model to a newly created ntuple
   # of name "NTuple", stored in kNTupleFileName.
   # Not to use:
   # ntuple = RNTupleWriter.Recreate(
   #                                  std.move(model),
   #                                  "NTuple",
   #                                  kNTupleFileName,
   #                                  ) # auto
   # Instead:
   global ntuple
   ntuple = RNTupleWriter_Recreate_Py(
                                       std.move( model ) ,    # data model
                                       "NTuple"          ,    # name
                                       kNTupleFileName   ,    # root file name
                                       ) # -> std.unique_ptr[RNTupleWriter]
   
   # Loops a, b, c. 
   global entries, std_threads
   entries = []
   std_threads = std.vector[ std.thread ]( )

   # a.
   for i in range(0, kNWriterThreads, 1) :
      entries.append( ntuple.CreateEntry() )


   # Step previous to step b.
   # Necessary for loading internal clang settings. TODO: Why? 
   #                                                      Without it, errors occur.
   def FillData2( x, y ):
      print( "FillData2 test." )
   entry = entries[0]
   f_Py( FillData2, entry, ntuple ) # FillData2
   f_Py( FillData, entry, ntuple )  # FillData

   # b.
   for i in range( 0, kNWriterThreads, 1 ) :
      entry = entries[i]
      t_i = std_thread_Py( FillData, entry, ntuple )
      std_threads = \
            std_vector_thread_push_back_Py( std_threads, std.move(t_i) )

   # c.
   for thread in std_threads:
      thread.join()

      

   # Writing ntuple to disk.
   # Not to use: 
   # ntuple.__destruct__() # unique_ptr[RNTupleWriter]
   # Instead: 
   RNTupleWriter_delete_Py( ntuple ) 
   

   # Special clean up: twice.
   # for potentially memory issue.
   ntuple.__destruct__() # unique_ptr[RNTupleWriter] # 1st time. Necessary.
   ntuple.__destruct__() # RNTupleWriter             # 2nd time. Necessary.
   model.__destruct__()
   model.__destruct__()
   std_threads.__destruct__()
   std_threads.__destruct__()
   

# For all events, make one histogram with
# only one of the written vectors.
#
# void
def Read() :

   # Get NTuple.
   global ntuple
   ntuple = RNTupleReader.Open("NTuple", kNTupleFileName) # auto

   # TODO(jblomer): The »inner name« of the vector 
   #                should become "vpx._0".
   global viewVpx
   viewVpx  = ntuple.GetView[ float ]( "vpx._0" )             # RNTupleView[float]
   # Note: 
   #      "vpx._0" is a name convention for 
   #      the entry in std.vector rather than the entire std.vector.
   #      This two lines are different in structure but represent the same data.
   #      ntuple.GetView[ float ]            ( "vpx._0" )  # -> RNTupleView[float]
   #      ntuple.GetView[ std.vector[float] ]( "vpx"    )  # -> RNTupleView[std.vector[float]]
   #      Their difference lies in performance: one loads the entire vector, the other 
   #      only a single part of it. 
   #      

   # If you need to get other columns from ntuple use:
   # view_id  = ntuple.GetView[ int   ]( "id" )                 # RNTupleView[int]
   # view_vpx = ntuple.GetView[ std.vector[ float ] ]( "vpx" )  # RNTupleView[std.vector[float]]
   # view_vpy = ntuple.GetView[ std.vector[ float ] ]( "vpy" )  # RNTupleView[std.vector[float]]
   # view_vpz = ntuple.GetView[ std.vector[ float ] ]( "vpz" )  # RNTupleView[std.vector[float]]


   gStyle.SetOptStat(0)

   # I.
   # Prepare canvas with two pads.
   global c1
   c1 = TCanvas( "c2",
                 "Multi-Threaded Filling Example",
                 200,
                 10,
                 1500,
                 500,
                 ) # -> TCanvas
   c1.Divide(2, 1)
   
   # I.a.
   # Set first histogram.
   c1.cd(1)
   global h
   h = TH1F("h", "This is the px distribution", 100, -4, 4)
   h.SetFillColor(48)
   # Fill first histogram.
   # Iterate through all values of "vpx" for all events.
   for i in viewVpx.GetFieldRange() :
      h.Fill( viewVpx( i ) )
   #   
   # Prevent the histogram from disappearing.
   h.DrawCopy()
   
   # I.b.
   c1.cd(2)
   global nEvents, viewId, hFillSequence
   # Get ntuple.
   nEvents = ntuple.GetNEntries() # int
   # Get sub-information from ntuple.
   viewId = ntuple.GetView[ std.uint32_t ]("id") # RNTupleView[int]
   # Set histogram.
   hFillSequence = TH2F( ""                        ,
                         "Entry Id vs Thread Id;"    # X;
                         "Entry Sequence Number;"    # Y;
                         "Filling Thread"          , # Title
                         100                       , # N_x
                         0                         , # from_x
                         nEvents                   , # to_x
                         100                       , # N_y
                         0                         , # from_y
                         kNWriterThreads + 1       , # to_y
                         ) # -> TH2F
   # II. Fill histogram.
   for i in ntuple.GetEntryRange() :
      hFillSequence.Fill( i, viewId( i ) )

   # III. Prevent the histogram from disappearing.
   hFillSequence.DrawCopy()
   

   # IV. Clean up.
   viewVpx.__destruct__() # 1th.
   viewId.__destruct__()  # 2nd.
   ntuple.__destruct__()  # 3rd.
   # Note :
   #        Preserve the order of destruction. 
   #        First columns, then the entire ntuple.
   #        Otherwise a segmentation fault will occur at 
   #        "RColumn::~RColumn()."



# void
def ntpl007_mtFill() :
   print( _GREEN + " >>> Calling 'ntpl007_mtFill()' " + 
          _RESET )
   Write()
   Read()
   


if __name__ == "__main__":
   ntpl007_mtFill()
