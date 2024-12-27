## \file
## \ingroup tutorial_ntuple
## \notebook
##
##
## Vector:
##         Write and read "STL" vectors with "RNTuple" class.
##         Adapted from the "hvector.cpp" tree tutorial.
##
##
## \macro_image
## \macro_code
##
## \date January 2025
## \author The ROOT Team
## \translator P. P.


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
# Functionality, interface, and data format is still subject to changes.
# Do not use for real data!

# root 7
# experimental classes
from ROOT.Experimental import (
                                RNTuple,
                                RNTupleModel,
                                RNTupleWriter,
                                RNTupleReader,

                                )

# classes
from ROOT import (
                   TCanvas,
                   TH1F,
                   TRandom,
                   TSystem,
                   )
# ctypes
from ctypes import (
                     c_double,
                     c_float,
                     c_int,
                     )

# globals
from ROOT import (
                   gStyle,
                   gSystem,
                   gPad,
                   gRandom,
                   gBenchmark,
                   gROOT,
                   gInterpreter,
                   )


# c-integration
from ROOT.gInterpreter import (
                                ProcessLine,
                                Declare,
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
# appropriately in PyRoot yet in ROOT < v6.32.
# We fix this issue by using the cppyy wrapper method.
import subprocess
command = ( " g++                              "
            "     -shared                      "
            "     -fPIC                        "
            "     -o ntpl002_vector_wrapper.so      "
            "     ntpl002_vector_wrapper.cpp        "
            "     `root-config --cflags --libs`"
            "     -fmax-errors=1"               ,
            )
# Then execute.
try :
   subprocess.run( command, shell=True )
except :
   raise RuntimeError( "ntpl002_vector_wrapper.cpp function not well compiled.")

# Then load `ntpl002_vector_wrapper.cpp` .
cppyy.load_library( "./ntpl002_vector_wrapper.so" )
cppyy.include     ( "./ntpl002_vector_wrapper.cpp" )
from cppyy.gbl import (
                        RNTupleWriter_Recreate_Py, 
                        RNTupleReader_Open_Py, 

                        RNTupleWriter_delete_Py,

                        )
# At last, the wrapper functions is ready.
print()



# Where to store the ntuple of this example.
kNTupleFileName = "ntpl002_vector.root"

# Update the histogram GUI every so many fills.
kUpdateGuiFreq = 1000

# Number of events to generate.
kNEvents = 25000



# Generate kNEvents with vectors in kNTupleFileName.
#
# void
def Write() :
   print( _GREEN + " >>> Calling 'Write()' " + 
          _RESET )

   # We create a pointer to an empty data model.
   model = RNTupleModel.Create() # auto
   
   # Creating fields of std.vector is the same as creating 
   # fields of simple types. Also, ".MakeField" method returns
   # a pointer of the given type.
   fldVpx   = model.MakeField[ std.vector[ float ] ]( "vpx"   ) # -> vector[float]
   fldVpy   = model.MakeField[ std.vector[ float ] ]( "vpy"   ) # -> vector[float]
   fldVpz   = model.MakeField[ std.vector[ float ] ]( "vpz"   ) # -> vector[float]
   fldVrand = model.MakeField[ std.vector[ float ] ]( "vrand" ) # -> vector[float]
   
   # We hand-over the data model to a newly created ntuple of name "F", 
   # which will be stored with the name kNTupleFileName.
   # In return, we get a pointer to an ntuple that after we will fill.
   # Not to use:
   # ntuple = RNTupleWriter.Recreate(std.move(model), "F", kNTupleFileName) # auto
   # Instead:
   ntuple = RNTupleWriter_Recreate_Py( 
                                       std.move(model),
                                       "F",
                                       kNTupleFileName,
                                       ) # auto
   
   global hpx
   hpx = TH1F("hpx", "This is the px distribution", 100, -4, 4)
   hpx.SetFillColor(48)
   
   global c1
   c1 = TCanvas("c1", "Dynamic Filling Example", 200, 10, 700, 500)
   
   gRandom.SetSeed()

   for i in range(0, kNEvents, 1):
      npx = int( gRandom.Rndm(1) * 15 )
      
      fldVpx.clear()
      fldVpy.clear()
      fldVpz.clear()
      fldVrand.clear()
      
      # Set the field data for the current event.
      for j in range(0, npx, 1):

         # - - -
         # Using "ctypes.c_float".
         # 
         # px = c_float() 
         # py = c_float() 
         # gRandom.Rannor(px, py)
         # pz = c_float( 
         #               px.value ** 2   + \
         #               py.value ** 2     \
         #               )
         #
         # random = gRandom.Rndm(1) # auto
         # 
         # hpx.Fill( px.value )
         # 
         # fldVpx.emplace_back   ( px     )
         # fldVpy.emplace_back   ( py     )
         # fldVpz.emplace_back   ( pz     )
         # fldVrand.emplace_back ( random )

         # - - -
         # Using "numpy.zeros".
         px = np.zeros( 1, dtype="float" )
         py = np.zeros( 1, dtype="float" )
         gRandom.Rannor(px, py)
         pz = px** 2 + py** 2    
         
         random = gRandom.Rndm(1) # auto
         
         hpx.Fill( px )
         
         fldVrand.emplace_back[ float ] ( random ) # 1st. Necessary!
         fldVpx.  emplace_back[ float ] ( px[0]  )
         fldVpy.  emplace_back[ float ] ( py[0]  )
         fldVpz.  emplace_back[ float ] ( pz[0]  )
         # Note: 
         #       If you use "numpy.zeros" with "std.vector.emplace_back",
         #       be sure to load a python-float number FIRST.
         #       Otherwise, ".emplace_back" will recognize it as float64
         #       and it'll generate error.
         #       However, you can use the template resolution method
         #       like "emplace_back[float]".
         #       This phenomenona doesn't occur with the ctypes method,
         #       but "numpy" is more practical.
         
      # GUI(Graphical User Interface)  updates.
      if (i and (i % kUpdateGuiFreq) == 0)  :
         if (i == kUpdateGuiFreq)  :
            hpx.Draw()
            
         c1.Modified()
         c1.Update()
         if (gSystem.ProcessEvents())  :
            break
            
      ntuple.Fill()
   
   hpx.DrawCopy()
   
   # Writing ntuple on disk by deleting the smart pointer 
   # using cpppy wrapper function.
   RNTupleWriter_delete_Py( ntuple )

   # Destruction of PyRoot object.
   ntuple.__destruct__() # 1. Smart pointer.
   ntuple.__destruct__() # 2. The pointer.

   print()


# For all events, make histogram only one of the written vectors.
#
# void
def Read() :
   print( _GREEN + " >>> Calling 'Read()' " +
          _RESET )

   # - - - 1 - - -
   #
   # Get a pointer to an empty RNTuple model.
   model = RNTupleModel.Create() # auto
   
   # We only define the fields that are needed for reading.
   fldVpx = model.MakeField[ std.vector[ float ] ]( "vpx" ) # auto
   
   # Create an ntuple without imposing a specific data model.  
   # Note: 
   #    We could generate the data model from the ntuple
   #    but instead here we prefer the view because we only want to
   #    access a single field.
   # Not to use:
   # ntuple = RNTupleWriter.Open(std.move(model), "F", kNTupleFileName) # auto
   # Instead:
   ntuple = RNTupleReader_Open_Py(
                                   std.move(model),
                                   "F",
                                   kNTupleFileName,
                                   ) # auto
   
   # Quick overview of the ntuple's key meta-data.
   print( _GREEN + " >>> Quick overview of the ntuple's key meta-data" +
          _RESET )
   ntuple.PrintInfo()
   print()
  
   # Entry number 42 in JSON format.
   print( _GREEN + " >>> Entry number 42 in JSON format:" +
          _RESET )
   ntuple.Show(41)
   print()

   # In a future version of RNTuple, there will be support for:
   # ntuple.Scan()

   
   # - - - 2 - - -
   #
   global c2, h
   c2 = TCanvas("c2", "Dynamic Reading Example", 200, 10, 700, 500) # TCanvas
   h = TH1F("h", "This is the px distribution", 100, -4, 4)
   h.SetFillColor(48)
   
   # Iterate through all events
   # using i as an event number and 
   # also as an index for accessing the view.
   for entryId in ntuple : 
      ntuple.LoadEntry( entryId )
      
      for px in fldVpx : 
         h.Fill( px )
         
      # Filter.
      if (entryId and (entryId % kUpdateGuiFreq) == 0)  :
         # Draw on condition.
         if (entryId == kUpdateGuiFreq)  :
            h.Draw()
            
         c2.Modified()
         c2.Update()

         # End loop.
         if (gSystem.ProcessEvents())  :
            break
         
   # Prevent the histogram from disappearing.
   h.DrawCopy()
   


# void
def ntpl002_vector() :
   Write()
   Read()
   


if __name__ == "__main__":
   ntpl002_vector()
