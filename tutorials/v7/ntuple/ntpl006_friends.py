## \file
## \ingroup tutorial_ntuple
## \notebook
##
## Friends: 
##          Work with a befriended "RNTuple" object.  
##          Adapted from "tree3.C".
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
                       vector,
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

                                )

# classes
from ROOT import (
                   TCanvas,
                   TH1F,
                   TMath,
                   TRandom,

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
            "     -o ntpl006_friends_wrapper.so      "
            "     ntpl006_friends_wrapper.cpp        "
            "     `root-config --cflags --libs`"
            "     -fmax-errors=1"               ,
            )
# Then execute.
try :
   subprocess.run( command, shell=True )
except :
   raise RuntimeError( "ntpl006_friends_wrapper.cpp function not well compiled.")

# Then load `ntpl006_friends_wrapper.cpp` .
cppyy.load_library( "./ntpl006_friends_wrapper.so" )
cppyy.include     ( "./ntpl006_friends_wrapper.cpp" )
from cppyy.gbl import (
                        RNTupleWriter_Recreate_Py, 
                        RNTupleReader_Open_Py, 

                        RNTupleWriter_delete_Py,
                        RNTupleReader_OpenFriends_Py,

                        )
# At last, the wrapper functions is ready.
print()



# Relevant constant names.
kNTupleMainFileName   = "ntpl006_data.root"
kNTupleFriendFileName = "ntpl006_reco.root"



# void
def Generate() :
   print( _GREEN + " >>> Calling 'Generate()' " + 
          _RESET )

   kMaxTrack = 500
   
   global modelData
   modelData = RNTupleModel.Create() # auto
   fldPx = modelData.MakeField[ std.vector[ float ] ]("px") # auto
   fldPy = modelData.MakeField[ std.vector[ float ] ]("py") # auto
   fldPz = modelData.MakeField[ std.vector[ float ] ]("pz") # auto
   
   modelFriend = RNTupleModel.Create() # auto
   fldPt = modelFriend.MakeField[ std.vector[ float ] ]("pt") # auto
   
   ntupleData = RNTupleWriter_Recreate_Py( 
                                           std.move(modelData),
                                           "data",
                                           kNTupleMainFileName,
                                           ) # auto
   ntupleReco = RNTupleWriter_Recreate_Py( 
                                           std.move(modelFriend),
                                           "reco",
                                           kNTupleFriendFileName,
                                           ) # auto
   
   for i in range(0, 1000, 1):
      ntracks = int( gRandom.Rndm() * (kMaxTrack - 1) ) # int
      for n in range(0, ntracks, 1):
         fldPx.emplace_back( gRandom.Gaus(0  , 1) )
         fldPy.emplace_back( gRandom.Gaus(0  , 2) )
         fldPz.emplace_back( gRandom.Gaus(10 , 5) )
         # TMath.Sqrt -> sqrt
         fldPt.emplace_back( 
                             sqrt( fldPx.at(n) * fldPx.at(n) + \
                                   fldPy.at(n) * fldPy.at(n)   \
                                   )
                             )
      ntupleData.Fill()
      ntupleReco.Fill()
      
      fldPx.clear()
      fldPy.clear()
      fldPz.clear()
      fldPt.clear()
   
   # Writing on disk is done automatically after 
   # ntuple goes out of scope.
   

# void
def ntpl006_friends() :
   print( _GREEN + " >>> Calling 'ntpl006_friends()' " + 
          _RESET )

   Generate()
   
   friends_ls = [ 
                  RNTupleReader.ROpenSpec( "data", kNTupleMainFileName   ),
                  RNTupleReader.ROpenSpec( "reco", kNTupleFriendFileName ),
                  ]
   friends_vec = std.vector[ RNTupleReader.ROpenSpec ]( friends_ls )

   # Not to use:
   # ntuple = RNTupleReader.OpenFriends( friends_vec ) # auto
   # Instead:
   ntuple = RNTupleReader_OpenFriends_Py( friends_vec ) # auto
   
   global c
   c = TCanvas("c", "", 200, 10, 700, 500) # auto # new
   h = TH1F("h", "pz {pt > 3.}", 100, -15, 35)
   
   viewPz = ntuple.GetView[ float ]("data.pz._0") # auto
   viewPt = ntuple.GetView[ float ]("reco.pt._0") # auto
   for i in viewPt.GetFieldRange() :
      if (viewPt(i) > 3.) :
         h.Fill( viewPz(i) )
         
      
   
   h.SetFillColor(48)
   h.DrawCopy()
   
   # Necessary to avoid memory leaks. 
   # Note: Double "__destruct__".
   viewPt.__destruct__()
   viewPz.__destruct__()
   ntuple.__destruct__()
   ntuple.__destruct__()



if __name__ == "__main__":
   ntpl006_friends()
