## \file
## \ingroup tutorial_ntuple
## \notebook
##
## LHCb Open Data: 
##     How to convert LHCb run 1 open data from a "TTree" to "RNTuple" object.
##
##     This tutorial illustrates data conversion for a simple, tabular data model.
##     For reading, the tutorial shows the use of an ntuple "View", which 
##     selectively accesses specific fields.
##     If a view is used for reading, there is no need to define the data model as 
##     an "RNTupleModel" first.
##     The advantage of a view is that it directly accesses "RNTuple"'s data 
##     buffers without making any additional memory copy.
## 
##
## \macro_image
## \macro_code
##
## \date April 2025
## \author The ROOT Team
## \translator P. P.


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
from ROOT import Experimental
from ROOT.Experimental import (
                                RField,
                                RNTuple,
                                RNTupleModel,
                                
                                RNTupleModel,
                                # Detail.RFieldBase, # error
                                RNTupleReader,
                                RNTupleWriter,

                                RDrawable,
                                RCanvas,
                                RColor,
                                RHistDrawable,
                                RNTuple,
                                RNTupleDS,
                                RNTupleModel,
                                RNTupleWriteOptions,
                                RNTupleWriter,
                                )
# Import classes from experimental namespace.
RFieldBase = Experimental.Detail.RFieldBase

# classes
from ROOT import (
                   TBranch,
                   TCanvas,
                   TFile,
                   TH1F,
                   TLeaf,
                   TTree,
                   )

# maths
from ROOT import (
                   sin,
                   cos,
                   sqrt,
                   )

# constants
from ROOT import (
                   kBlack,
                   kOrange,
                   kViolet,
                   kSpring,
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
            "     -o ntpl003_lhcbOpenData_wrapper.so      "
            "     ntpl003_lhcbOpenData_wrapper.cpp        "
            "     `root-config --cflags --libs`"
            "     -fmax-errors=1"               ,
            )
# Then execute.
try :
   subprocess.run( command, shell=True )
except :
   raise RuntimeError( "ntpl003_lhcbOpenData_wrapper.cpp function not well compiled.")

# Then load `ntpl003_lhcbOpenData_wrapper.cpp` .
cppyy.load_library( "./ntpl003_lhcbOpenData_wrapper.so" )
cppyy.include     ( "./ntpl003_lhcbOpenData_wrapper.cpp" )
from cppyy.gbl import (
                        RNTupleWriter_Recreate_Py, 
                        RNTupleReader_Open_Py, 

                        RNTupleWriter_delete_Py,

                        RFieldBase_Create_Unwrap_Py,
                        REntry_GetRawPtr_Py,
                        RNTupleModel_AddField_Py,

                        RNTupleReader_GetView_int_Py,
                        RNTupleReader_GetView_float_Py,
                        RNTupleReader_GetView_double_Py,

                        )
# At last, the wrapper functions is ready.
print()



# Relevant constant names.
kTreeFileName   = "https://root.cern.ch/files/LHCb/lhcb_B2HHH_MagnetUp.root"
kNTupleFileName = "ntpl003_lhcbOpenData.root"



# void
def Convert() :
   print( _GREEN + " >>> Calling 'Convert()' " + 
          _RESET )

   f = TFile.Open(kTreeFileName)     # -> TDavixFile
   assert( f and not f.IsZombie() ), f"Imposible to reach {kTreeFileName}"

   
   # Get a pointer to an empty RNTuple model
   # without a default entry.
   model = RNTupleModel.CreateBare() # -> RNTupleModel
   
   # We create RNTuple fields based on the types found in the TTree.
   # This simple approach only works for trees with simple branches 
   # and only one leaf per branch.
   tree = f.Get( "DecayTree" ) # -> TTree
   for b in tree.GetListOfBranches() : 
      # Testing each branch of "DecayTree".
      assert(b)
      
      # We assume every branch has a single leaf.
      l = b.GetListOfLeaves().First() # -> TLeafI
      
      # Create an ntuple field with the same name and type 
      # of the tree branch.
      # Not to use:
      # field = RFieldBase_Create_Py( ... ).Unwrap()
      # Instead:
      field = RFieldBase_Create_Unwrap_Py( 
                                           l.GetName(),
                                           l.GetTypeName(),

                                           ) # -> RFieldBase *
      print( "Convert leaf " ,
             l.GetName()     ,
             " ["            ,
             l.GetTypeName() ,
             "]"             ,
             )
      print( " --> "         ,
             )
      print( "field "        ,
             field.GetName() ,
             " ["            ,
             field.GetType() ,
             "]"             ,
             )
      
      # Hand over ownership of the field to the ntuple model.  
      # This will also create a memory location attached
      # to the model's default entry, that will be used 
      # to place the data supposed to be written.
      # Not to use:
      # model.AddField( std.move( field ) )
      # Instead:
      model = RNTupleModel_AddField_Py( std.move( model ), std.move( field ) )

      
   
   # The new ntuple takes ownership of the model.
   ntuple = RNTupleWriter_Recreate_Py(
                                       std.move(model),
                                       "DecayTree",
                                       kNTupleFileName,
                                       ) # -> RNTupleWriter *
   
   entry = ntuple.GetModel().CreateEntry() # -> auto
   for b in tree.GetListOfBranches() :
      l = b.GetListOfLeaves().First() # -> TLeaf 

      # We connect the model's default entry's memory location
      # for the new field to the branch, so that we can
      # fill the ntuple with the data read from the TTree.
      # Not to use:
      # fieldDataPtr = entry.GetRawPtr( l.GetName() ) # void *
      # Instead:
      fieldDataPtr = REntry_GetRawPtr_Py( entry, l.GetName() )
      tree.SetBranchAddress( b.GetName(), fieldDataPtr )
 
   

   print( _GREEN + " >>> Getting entries ... to write on disk. " + 
          _RESET )
   nEntries = tree.GetEntries() # -> int
   for i in range( 0, nEntries, 1 ) :      # ok. 
   # for i in range( 0, 10, 1 ) :          # ok.
   # for i in range( 0, 4*100000, 1 ) :    # ok.
      
      tree.GetEntry( i )
      ntuple.Fill( entry )
      
      # Status report by million.
      if (i and i % 100000 == 0) :
         print( "Wrote "   ,
                i          ,
                " entries" ,
                )
   

   print( _GREEN + " >>> Write on disk done. " + 
          _RESET )
   # Writing on disk.
   RNTupleWriter_delete_Py( ntuple ) 
   print()

   # Cleaning up.
   field.__destruct__()
   ntuple.__destruct__()
   model.__destruct__()
   f.__destruct__()


      
   

# void
def ntpl003_lhcbOpenData() :
   print( _GREEN + " >>> Calling 'ntpl003_lhcbOpenData()' " + 
          _RESET )

   Convert()
   
   # Create histogram of the flight distance.
   
   # We open the ntuple without specifying an explicit model first, 
   # but instead use a view on the field we are interested in:
   ntuple = RNTupleReader_Open_Py("DecayTree", kNTupleFileName) # auto

   
   # The view wraps a read-only double value and accesses.
   # directly the ntuple's data buffers.
   # Not to use:
   # viewFlightDistance = ntuple.GetView["double"]("B_FlightDistance") # auto
   # Instead:
   viewFlightDistance = RNTupleReader_GetView_double_Py( ntuple, "B_FlightDistance" ) # auto

   
   global c, h
   c = TCanvas("c", "B Flight Distance", 200, 10, 700, 500) # auto # new
   h = TH1F("h", "B Flight Distance", 200, 0, 140)
   h.SetFillColor(48)
   
   for i in ntuple.GetEntryRange() :
      # Note that we do not load an entry in this loop;
      # i.e. we avoid the memory copy of loading the data into
      # the memory location given by the entry.
      h.Fill( viewFlightDistance( i ) )
      
   
   h.DrawCopy()
   


if __name__ == "__main__":
   ntpl003_lhcbOpenData()
