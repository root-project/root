## \file
## \ingroup tutorial_ntuple
## \notebook
##
## Staff:
##        Write and read tabular data with "RNTuple" class.
##        Adapted from the cernbuild and cernstaff tree tutorials.
##        Illustrates the type-safe ntuple model interface, which
##        is used to define a data model that is in a second step
##        taken by an ntuple reader or writer.
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

                       ifstream,
                       )

# NOTE: The RNTuple classes are experimental at this point.
# Functionality, interface, and data format is still subject to changes.
# Do not use for real data!

# root 7
# experimental classes
from ROOT.Experimental import (
                                RNTupleModel,
                                RNTupleReader,
                                RNTupleWriter,

                                RNTuple,
                                RNTupleModel,

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
                   TCanvas,
                   TH1I,
                   TROOT,
                   TString,

                   TH1D,
                   TLegend,
                   TSystem,
                   TCanvas,
                   TFile,
                   TH1F,
                   TGraph,
                   TLatex,
                   TCanvas,
                   TPaveLabel,
                   TPavesText,
                   TPaveText,
                   TText,
                   TArrow,
                   TWbox,
                   TPad,
                   TBox,
                   TPad,
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
            "     -o ntpl001_staff_wrapper.so      "
            "     ntpl001_staff_wrapper.cpp        "
            "     `root-config --cflags --libs`"
            "     -fmax-errors=1"               ,
            )
# Then execute.
try :
   subprocess.run( command, shell=True )
except :
   raise RuntimeError( "ntpl001_staff_wrapper.cpp function not well compiled.")

# Then load `ntpl001_staff_wrapper.cpp` .
cppyy.load_library( "./ntpl001_staff_wrapper.so" )
cppyy.include     ( "./ntpl001_staff_wrapper.cpp" )
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

                        std_getline_Py,

                        )
# At last, the wrapper functions is ready.
print()



# Relevant name constants.
kNTupleFileName = "ntpl001_staff.root"



# void
def Ingest() :
   print( _GREEN + " >>> Calling 'Ingest()' " + 
          _RESET )

   # The input file cernstaff.dat is a copy of the CERN staff
   # data base from 1988.
   fin = ifstream( gROOT.GetTutorialsDir() + "/tree/cernstaff.dat" )
   assert( fin.is_open() )
   
   # We create a unique pointer to an empty data model.
   model = RNTupleModel.Create() # auto
   
   # To define the data model, we create fields with a given C++ type and name.  
   # Fields are roughly TTree branches.
   # MakeField returns a shared pointer to a memory location
   # that we can populate to fill the "ntuple" with data.
   fldCategory = model.MakeField[ int ]("Category") # auto
   fldFlag     = model.MakeField[ int ]("Flag") # auto unsigned
   fldAge      = model.MakeField[ int ]("Age") # auto
   fldService  = model.MakeField[ int ]("Service") # auto
   fldChildren = model.MakeField[ int ]("Children") # auto
   fldGrade    = model.MakeField[ int ]("Grade") # auto
   fldStep     = model.MakeField[ int ]("Step") # auto
   fldHrweek   = model.MakeField[ int ]("Hrweek") # auto
   fldCost     = model.MakeField[ int ]("Cost") # auto
   fldDivision = model.MakeField[ std.string ]("Division") # auto
   fldNation   = model.MakeField[ std.string ]("Nation") # auto
   
   # We hand-over the data model to a newly created ntuple of 
   # name "Staff", stored with the name kNTupleFileName.
   # In return, we get a pointer to an ntuple that
   # we can fill.
   # Not to use:
   # ntuple = RNTupleWriter.Recreate( std.move(model),
   #                                  "Staff",
   #                                  kNTupleFileName,
   #                                  ) # auto
   # Instead:
   ntuple = RNTupleWriter_Recreate_Py( std.move(model),
                                       "Staff",
                                       kNTupleFileName,
                                       ) # auto
   
   # record = ctypes.create_string_buffer(100)
   # std_getline_Py(fin, record, len(record) + 12 )

   record = std.string()
   while ( std.getline(fin, record) ) :
      dummy = str( record ).split()
      fldCategory .get () [0] = int( dummy[ 0  ] ) # int
      fldFlag     .get () [0] = int( dummy[ 1  ] ) # ...
      fldAge      .get () [0] = int( dummy[ 2  ] )
      fldService  .get () [0] = int( dummy[ 3  ] )
      fldChildren .get () [0] = int( dummy[ 4  ] )
      fldGrade    .get () [0] = int( dummy[ 5  ] )
      fldStep     .get () [0] = int( dummy[ 6  ] )
      fldHrweek   .get () [0] = int( dummy[ 7  ] ) # ...
      fldCost     .get () [0] = int( dummy[ 8  ] ) # int
      fldDivision             =      dummy[ 9  ]   # std.string
      fldNation               =      dummy[ 10 ]   # std.string

      ntuple.Fill()

      
   
   # The ntuple unique pointer goes out of scope here.
   # In PyRoot, however, at the time of destruction, the ntuple needs to
   # deletion of "C" so that it flushes out unwritten data to disk
   # and closes the attached ROOT file.
   RNTupleWriter_delete_Py( ntuple )

   print()
   


# void
def Analyze() :
   print( _GREEN + " >>> Calling 'Analyze()' " + 
          _RESET )

   # Get a unique pointer to an empty RNTuple model.
   model = RNTupleModel.Create() # auto )
   
   # We only define the fields that are needed for reading.
   # std.shared_ptr[int] fldAge = model.MakeField[ int ]("Age")
   fldAge = model.MakeField[ int ]("Age") # std.shared_ptr[int]
   
   # Create an ntuple and attach the read model to it.
   ntuple = RNTupleReader_Open_Py( std.move(model),
                                   "Staff",
                                   kNTupleFileName,
                                   ) # auto
   
   # Quick overview of the ntuple and list of fields.
   print( _GREEN + " >>> Quick overview of the ntuple and list of fields." +
          _RESET )
   ntuple.PrintInfo()
   print()
   
   print( _GREEN + " >>> The first entry in JSON format:" +
          _RESET )
   ntuple.Show(0)
   print()

   # In a future version of RNTuple, there will be support for:
   # ntuple.Scan()
   
   global c
   c = TCanvas("c", "", 200, 10, 700, 500) # auto # new
   h = TH1I("h", "Age Distribution CERN, 1988", 100, 0, 100)
   h.SetFillColor(48)
   
   # for (auto entryId : *ntuple)  :
   for entryId in ntuple: 
      # Populate fldAge
      ntuple.LoadEntry( entryId )
      h.Fill( fldAge.get()[0] )
      
   
   h.DrawCopy()
   

# void
def ntpl001_staff() :
   Ingest()
   Analyze()
   


if __name__ == "__main__":
   ntpl001_staff()
