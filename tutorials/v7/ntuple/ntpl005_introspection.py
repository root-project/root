## \file
## \ingroup tutorial_ntuple
## \notebook
##
## Introspection:
## This tutorial shows how to write
## and read an "RNTuple" object from a user-defined class.
## Also, this illustrates various "RNTuple" introspection methods.
## It was Adapted from "tv3.C".
##
## Keywords:
## introspection, rntuple, struct
## 
##
## \macro_image
## \macro_code
##
## \date January 2025 
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
                       move,
                       )


# NOTE: The RNTuple classes are experimental at this point.
# Functionality, interface, and data format is still subject to changes.
# Do not use them for real data!

# root 7
# experimental classes
from ROOT import Experimental
from ROOT.Experimental import (
                                ENTupleInfo,
                                RNTupleModel,
                                RNTupleReader,
                                RNTupleWriter,
                                RNTupleWriteOptions,

                                )
# classes
from ROOT import (
                   TCanvas,
                   TH1,
                   TH1F,
                   TRandom,
                   TSystem,
                   TH1D,
                   TLegend,
                   TSystem,
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
# appropriately in PyRoot.
# We fix this issue by using the cppyy wrapper method.
import subprocess
command = ( " g++                              "
            "     -shared                      "
            "     -fPIC                        "
            "     -o ntpl005_introspection_wrapper.so      "
            "     ntpl005_introspection_wrapper.cpp        "
            "     `root-config --cflags --libs`"
            "     -fmax-errors=1"               ,
            )
# Then execute.
try :
   subprocess.run( command, shell=True )
except :
   raise RuntimeError( "ntpl005_introspection_wrapper.cpp function not well compiled.")

# Then load `ntpl005_introspection_wrapper.cpp` .
cppyy.load_library( "./ntpl005_introspection_wrapper.so" )
cppyy.include     ( "./ntpl005_introspection_wrapper.cpp" )
from cppyy.gbl import (
                        RNTupleWriter_Recreate_Py, 
                        RNTupleReader_Open_Py,
                        # REntry_Get,  # ... into two specific functions.
                        REntry_Get_vector_float_Py,  # 1.
                        REntry_Get_uint32_Py,        # 2.

                        std_thread_Py,
                        std_vector_thread_push_back_Py,

                        RNTupleWriter_delete_Py,
                        f_Py, # TODO: 
                           #       Review function signature arguments 
                           #       for extern "C" wrappers.
                           #       Here "f" is a :f

                        )
# At last, the wrapper functions is ready.
print()



# Relevant constants.
kNTupleFileName = "ntpl005_introspection.root"



# Store entries of type Vector3 in the ntuple
class Vector3_Py :
   # private:
   def __init__( self, ) :
      self.fX = 0
      self.fY = 0
      self.fZ = 0
   
   #public:
   def x( self, ) :
      return self.fX
      
   def y( self, ) :
      return self.fY
      
   def z( self, ) :
      return self.fZ
      
   
   def SetXYZ( self, x : float, y : float, z : float) :
      self.fX = x
      self.fY = y
      self.fZ = z

   # TODO: Review.
   # for Vector3.SetXYZ( Vector3, 1,1,1) # Vector3.fX = 1 ,... 
   fX = 0
   fY = 0
   fZ = 0

# C class defition of Vector3.
ProcessLine("""
            // Store entries of type Vector3 in the ntuple
            class Vector3 {
             private:
               double fX = 0;
               double fY = 0;
               double fZ = 0;
            
             public:
               double x() const {
                  return fX;
               }
               double y() const {
                  return fY;
               }
               double z() const {
                  return fZ;
               }
            
               void SetXYZ(double x, double y, double z) {
                  fX = x;
                  fY = y;
                  fZ = z;
               }
            };
            """)
Vector3 = ROOT.Vector3


# void
def Generate() :

   model = RNTupleModel.Create() # auto
   fldVector3 = model.MakeField[ Vector3 ]("v3") # auto
   
   # Explicitly enforce a certain compression algorithm.
   options = RNTupleWriteOptions()
   options.SetCompression( ROOT.RCompressionSetting.EDefaults.kUseGeneralPurpose )
   
   # Not to use:
   # ntuple = RNTupleWriter.Recreate( std.move(model),
   # Instead:
   ntuple = RNTupleWriter_Recreate_Py( std.move(model),
                                       "Vector3",
                                       kNTupleFileName,
                                       options
                                       ) # auto
   r = TRandom()
   # for (unsigned int i = 0; i < 500000; ++i)  :
   for i in range(0, 500000, 1) : 
      fldVector3.SetXYZ( r.Gaus(0, 1),
                         r.Landau(0, 1),
                         r.Gaus(100, 10),
                         )
      ntuple.Fill()

   # "ntuple" will go out of scope. But using the method __del__,
   # which doesn't have implemented a proper of destruct it.
   # Use momentarely this to write "ntuple" on disk:
   RNTupleWriter_delete_Py( ntuple )
   

# TODO
# Context Manager for GetView method so that
# we can use "with ntuple.GetView... as myView: "
def RNTupleReader_GetView__enter__(self, ):
   pass
def RNTupleReader_GetView__exit__(self, ):
   # viewX.__destruct__() TODO
   pass
# RNTupleReader.GetView.__enter__ = RNTupleReader_GetView__enter__
# RNTupleReader.GetView.__exit__  = RNTupleReader_GetView__exit__




# void
def ntpl005_introspection() :

   Generate()
   
   global ntuple
   ntuple = RNTupleReader_Open_Py( "Vector3", kNTupleFileName ) # auto
   
   # Display the schema of the ntuple.
   print( _GREEN + " >>> Display the schema of the ntuple." +
          _RESET )
   ntuple.PrintInfo()
   
   # Display information about the storage layout of the data.
   print( _GREEN + " >>> Display information about "   + 
                   "the storage layout of the data."   +
          _RESET )
   ntuple.PrintInfo( ENTupleInfo.kStorageDetails )
   
   # Display the first entry.
   print( _GREEN + " >>> Display the first entry." +
          _RESET )
   ntuple.Show(0)
   print()
   
   # Collect I/O runtime counters when processing the data set.
   # Maintaining the counters comes with a small performance 
   # overhead, so it has to be explicitly enabled.
   ntuple.EnableMetrics()
   
   # Plot the y components of vector3
   global c1
   c1 = TCanvas("c1", "RNTuple Demo", 10, 10, 600, 800) # TCanvas
   c1.Divide(1, 2)
   c1.cd(1)
   global h1
   h1 = TH1F("x", "x component of Vector3", 100, -3, 3)

   # We enclose viewX in a scope in order to indicate to
   # the RNTuple when we are not anymore interested
   # in v3.fX .
   # TODO : "with" protocol is not fully pythonized yet for ".GetView" method.
   # with ntuple.GetView[ float ]( "v3.fX" ) as viewX : # error. ->  float != SplitReal64
   # with ntuple.GetView[ "double" ]( "v3.fX" ) as viewX : # ok. -> "double"->No context manager.
   
   from contextlib import nullcontext
   with nullcontext() as x: 
      # TODO: 
      #       "viewX" definition and destruction. 
      #       Redundant but necessary for future versions when ".GetView" 
      #       become fully pythonized.
      viewX = ntuple.GetView[ "double" ]( "v3.fX" )
      for i in ntuple.GetEntryRange() :
         h1.Fill( viewX( i ) )
         
      # Necessary until ".GetView" gets pythonized.
      viewX.__destruct__()
         
      
   h1.DrawCopy()
   
   c1.cd(2)
   global h2
   h2 = TH1F("y", "y component of Vector3", 100, -5, 20)
   viewY = ntuple.GetView[ "double" ]( "v3.fY" ) # auto
   for i in ntuple.GetEntryRange() :
      h2.Fill( viewY( i ) )
      
   h2.DrawCopy()
   
   # Display the I/O operation statistics performed by the RNTuple reader.
   print( _GREEN + " >>> Display the I/O operation "              +
                   "statistics performed by the RNTuple reader."  +
          _RESET )
   ntuple.PrintInfo( ENTupleInfo.kMetrics )
   print()
   
   # We read 2 out of the 3 Vector3 members and thus
   # we should have requested approximately 2/3 of the file.
   fileStat = ROOT.FileStat_t()
   retval = gSystem.GetPathInfo(kNTupleFileName, fileStat) # auto
   assert(retval == 0)
   fileSize = float( fileStat.fSize )
   _szReadPayload  = "RNTupleReader.RPageSourceFile.szReadPayload"
   _szReadOverhead = "RNTupleReader.RPageSourceFile.szReadOverhead"
   nbytesRead = \
      ntuple.GetMetrics().GetCounter( _szReadPayload  ).GetValueAsInt()  +  \
      ntuple.GetMetrics().GetCounter( _szReadOverhead ).GetValueAsInt()     \
      # -> float
  
   print( _GREEN + " >>> FINAL 'ntpl005_introspection()' OUTPUT: " +
          _RESET )
   print( "File size:      "                     ,
          round( fileSize / 1024. / 1024. , 2)   ,
          " MiB"                                 ,
          )
   print( "Read from file: "                     ,
          round( nbytesRead / 1024. / 1024. , 2) ,
          " MiB"                                 ,
          )
   print( "Ratio:          "                     ,
          round( nbytesRead / fileSize , 2)      ,
          )
   


if __name__ == "__main__":
   ntpl005_introspection()
