## \file
## \ingroup tutorial_ntuple
## \notebook
##
## Import:
##         Example of converting data stored
##         in a "TTree" into an "RNTuple".
##
## \macro_image
## \macro_code
##
## \date January 2025
## \author The ROOT Team
## \translator P. P.


import ROOT
import cppyy


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
                                RNTupleImporter,
                                RNTupleReader,

                                RNTuple, 
                                RNTupleDS, 
                                RNTupleImporter, 

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

# classes
from ROOT import (
                   TFile,
                   TROOT,

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
# Define color codes
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
            "     -o ntpl008_import_wrapper.so      "
            "     ntpl008_import_wrapper.cpp        "
            "     `root-config --cflags --libs`"
            "     -fmax-errors=1"               ,
            )
# Then execute.
try :
   subprocess.run( command, shell=True )
except :
   raise RuntimeError( "ntpl008_import_wrapper.cpp function not well compiled.")

# Then load `ntpl008_import_wrapper.cpp` .
cppyy.load_library( "./ntpl008_import_wrapper.so" )
cppyy.include     ( "./ntpl008_import_wrapper.cpp" )
from cppyy.gbl import (
                        RNTupleWriter_Recreate_Py,
                        RNTupleReader_Open_Py,

                        RNTupleWriter_delete_Py

                        )
# At last, the wrapper functions is ready.



# Relevant constant names. 
kTreeFileName   = \
      "http://root.cern.ch/files/HiggsTauTauReduced/GluGluToHToTauTau.root"
kTreeName       = "Events"
kNTupleFileName = "ntpl008_import.root"



# void
def ntpl008_import() :
   print( _GREEN + " >>> Calling 'ntpl008_import()' " + 
          _RESET )

   # RNTupleImporter appends keys to the output file; 
   # make sure a second run of the tutorial does not fail with
   # `Key 'Events' already exists in file ntpl008_import.root` 
   # by removing the output file.
   ROOT.gSystem.Unlink( kNTupleFileName )
   
   # Use multiple threads to compress RNTuple data.
   ROOT.EnableImplicitMT()
   
   # Create a new RNTupleImporter object.
   importer = RNTupleImporter.Create(
                                      kTreeFileName,
                                      kTreeName,
                                      kNTupleFileName,
                                      ) # auto
   
   print( _GREEN + " >>> Begin importing." +
          _RESET )
   # Begin importing.
   importer.Import()
   print()
   
   
   # Inspect the schema of the written RNTuple.
   print( _GREEN + " >>> Inspect the schema of the written RNTuple." +
          _RESET )
   file = TFile.Open( kNTupleFileName ) # auto
   if ( not file or file.IsZombie() ) :
      raise RuntimeError( "cannot open " , kNTupleFileName )
   print()
      
   # Getting and Reading RNTuple 'Events' .
   print( _GREEN + " >>> Getting and Reading RNTuple 'Events' ." +
          _RESET )
   # Not to use:
   # ntpl = file.Get[ RNTuple ]("Events") # auto
   # Instead:
   ntpl = file.Get("Events") # -> RNTuple 
   reader = RNTupleReader.Open( ntpl ) # auto
   print()

   print( _GREEN + " >>> Request reader info." +
          _RESET )
   reader.PrintInfo()
   print()
   
   df = ROOT.RDF.Experimental.FromRNTuple(
                                           "Events",
                                           kNTupleFileName,
                                           ) # auto
   # Not to use:
   # df.Histo1D( 
   #             ["Jet_pt",
   #              "Jet_pt",
   #              100,
   #              0,
   #              0,
   #              "Jet_pt",
   #              ], # ->TH1DModel
   #            ).DrawCopy()
   # Instead:
   global h1d
   h1d = ROOT.RDF.TH1DModel( 
                             "Jet_pt",
                             "Jet_pt",
                             100,
                             0,
                             0,
                             )
   df.Histo1D(
               h1d,
               "Jet_pt",
               ).DrawCopy()
   
   


if __name__ == "__main__":
   ntpl008_import()
