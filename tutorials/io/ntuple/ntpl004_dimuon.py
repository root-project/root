## \file
## \ingroup tutorial_ntuple
## \notebook
## 
## Dimuon: 
##        A mini-analysis of CMS OpenData with "RDataFrame" class.
##        This tutorial illustrates that 
##        analyzing data with "RDataFrame" class works the same way
##        as with "TTree" class data and "RNTuple" class data.  
## 
## In here, the used "RNTuple" class data are converted from the 
## tree events located in:
## http://root.cern.ch/files/NanoAOD_DoubleMuon_CMS2011OpenData.root
##
## Based on RDataFrame's df102_NanoAODDimuonAnalysis.py tutorial.
##
##
## \macro_image
## \macro_code
##
## \date January 2025
## \author The ROOT Team
## \translator P. P.


import cppyy
import ctypes
import ROOT


# standard library
from ROOT import std
from ROOT.std import (
                       make_shared,
                       unique_ptr,
                       # cassert, # error
                       # cmath, # error
                       iostream,
                       # memory, # error
                       string,
                       # utility, # error
                       vector,
                       )


# standard c library
libc = ctypes.CDLL("libc.so.6")
# sscanf
sscanf = libc.sscanf


# Important.
# NOTE: The RNTuple classes are experimental at this point.
#       Functionality, interface, and data format is still subject to changes.
#       Do not use for real data!
#       Feedback is welcome.

# root 7
# experimental classes
from ROOT import Experimental
from ROOT.Experimental import (
                                RNTupleReader,
                                RNTupleDS,
                                # RDataFrame, # error, not import like this.
                                RNTuple,
                                RNTupleDS,
                                # RVec, # error
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

#
from ROOT import RDataFrame
from ROOT.RDF import (
                       TH1DModel,
                       )
# Not to use:
#from ROOT.RDF.Experimental import (
#                                    FromRNTuple,
#                                    )
# Instead:
from ROOT.RDF import Experimental as RDF_Experimental 
FromRNTuple = RDF_Experimental.FromRNTuple



# classes
from ROOT import (
                   RVec,
                   TCanvas,
                   TH1D,
                   TLatex,
                   TStyle,

                   )

# maths
from ROOT import (
                   sin,
                   cos,
                   sqrt,
                   )
# ctypes
from ctypes import (
                     c_double,
                     c_int,
                     c_uint,
                     c_float,
                     c_char_p,
                     c_void_p,
                     create_string_buffer,
                     #
                     byref,
                     POINTER,
                     cast,
                     )
# ctypes
c_string = create_string_buffer


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
            "     -o ntpl004_dimuon_wrapper.so      "
            "     ntpl004_dimuon_wrapper.cpp        "
            "     `root-config --cflags --libs`"
            "     -fmax-errors=1"               ,
            )
# Then execute.
try :
   subprocess.run( command, shell=True )
except :
   raise RuntimeError( "ntpl004_dimuon_wrapper.cpp function not well compiled.")

# Then load `ntpl004_dimuon_wrapper.cpp` .
cppyy.load_library( "./ntpl004_dimuon_wrapper.so" )
cppyy.include     ( "./ntpl004_dimuon_wrapper.cpp" )
from cppyy.gbl import (
                        RNTupleWriter_Recreate_Py, 
                        RNTupleReader_Open_Py, 

                        RFieldBase_Create_Unwrap_Py,
                        RNTupleModel_AddField_Py,

                        RNTupleReader_GetView_int_Py,
                        RNTupleReader_GetView_float_Py,
                        RNTupleReader_GetView_double_Py,

                        RInterface_Define_Py,

                        )
# At last, the wrapper functions are ready.
print()



# Relevant constants names.
kNTupleFileName = \
   "http://root.cern.ch/files/tutorials/ntpl004_dimuon_v1rc1.root"



# void
def ntpl004_dimuon() :
   print( _GREEN + " >>> Calling 'ntpl004_dimuon()' " + 
          _RESET )

   # Use all available CPU cores.
   ROOT.EnableImplicitMT()
   
   df = ROOT.RDF.Experimental.FromRNTuple("Events", kNTupleFileName) # auto
   
   # The tutorial is identical to df102_NanoAODDimuonAnalysis 
   # except in the use of
   # InvariantMassStdVector instead of InvariantMass.
   # To be fixed in a later version of RNTuple. Not in versions ROOT < 6.32.
   
   # For simplicity, select only events with exactly two muons.
   df_2mu = df.Filter( 
                       "#Muon == 2",                      # Condition.
                       "Events with exactly two muons",   # Title.
                       ) # auto
   # And select only those muons with opposite charges.
   df_os = df_2mu.Filter( 
                          "Muon.charge[0] != Muon.charge[1]",  # Condition.
                          "Muons with opposite charge",        # Title.
                          ) # auto
   
   # Compute invariant mass of the dimuon system.
   # Not to use: 
   # Not supported yet for versions < ROOT 6.30
   # df_mass = df_os.Define( 
   #                         "Dimuon_mass",         #  Name.
   #                         ROOT.VecOps.InvariantMass[float],  # Callable.
   #                         [                     # Parameters:
   #                           "Muon.pt",          # Parameter 1 of Callable.
   #                           "Muon.eta",         # Parameter 2 of Callable.
   #                           "Muon.phi",         # Parameter 3 of Callable.
   #                           "Muon.mass",        # Parameter 4 of Callable.
   #                           ],
   #                         ) # auto
   # Instead:
   # For simple calculations, you could use:
   # global df_mass
   # df_mass = df_os.Define( 
   #                         "Dimuon_mass",      # Name.
   #                         "Muon.pt + Muon.eta + Muon.phi + Muon.mass", # Calc.
   #                         ) # auto
   # For complex calculations like InvariantMass, you could use:
   df_mass = RInterface_Define_Py( df_os,            # Pointer to DataFrame model.
                                   "Dimuon_mass",    # Name.
                                   ROOT.VecOps.InvariantMass[float], # Callable.
                                   [                      
                                     "Muon.pt",      # Parameter 1 of Callable.
                                     "Muon.eta",     # Parameter 2 of Callable.
                                     "Muon.phi",     # Parameter 3 of Callable.
                                     "Muon.mass",    # Parameter 4 of Callable.
                                     ],
                                   )
   # Note:
   #      The above only works for InvariantMass callable.
   #      For different callables like other functions inside ROOT.VecOps,
   #      you should change the "InvariantMass_CFuncPtr" signature
   #      in the "ntpl004_dimuon_wrapper.cpp" file.
    
     
   # Make histogram of dimuon mass spectrum.
   # Not to use:
   #             Not supported yet for versions < ROOT 6.30
   # h = df_mass.Histo1D( 
   #                      [
   #                        "Dimuon_mass",
   #                        "Dimuon_mass",
   #                        30000,
   #                        0.25,
   #                        300,
   #                        ], # -> TH1DModel
   #                      "Dimuon_mass",
   #                      ) # auto
   # Instead:
   global h1d
   h1d = ROOT.RDF.TH1DModel( "Dimuon_mass",
                             "Dimuon_mass",
                             30000,
                             0.25,
                             300,
                             )
   global h
   h = df_mass.Histo1D( 
                        h1d,
                        "Dimuon_mass",
                        ) # auto
   
   # Request cut-flow report.
   print( _GREEN + " >>> Request cut-flow report." +
          _RESET )
   report = df_mass.Report() # auto
   print()
   
   # Produce plot.
   print( _GREEN + " >>> Produce plot " +
          _RESET )
   # a. Set canvas.
   gStyle.SetOptStat(0)
   gStyle.SetTextFont(42)
   global c
   c =  TCanvas("c", "", 800, 700) # auto # new
   c.SetLogx()
   c.SetLogy()
   
   # b. Set plot.
   h.SetTitle("")
   h.GetXaxis().SetTitle("m_{#mu#mu} (GeV)")
   h.GetXaxis().SetTitleSize(0.04)
   h.GetYaxis().SetTitle("N_{Events}")
   h.GetYaxis().SetTitleSize(0.04)
   # c. Draw plot.
   h.DrawCopy()
   
   # d. Set names of labels with their positions.
   global label
   label = TLatex()
   label.SetNDC(True)
   label.DrawLatex(0.175, 0.740, "#eta")
   label.DrawLatex(0.205, 0.775, "#rho,#omega")
   label.DrawLatex(0.270, 0.740, "#phi")
   label.DrawLatex(0.400, 0.800, "J/#psi")
   label.DrawLatex(0.415, 0.670, "#psi'")
   label.DrawLatex(0.485, 0.700, "Y(1,2,3S)")
   label.DrawLatex(0.755, 0.680, "Z")
   # e. Change size for two labels. 
   label.SetTextSize(0.040)
   label.DrawLatex(0.100, 0.920, "#bf{CMS Open Data}")
   label.SetTextSize(0.030)
   label.DrawLatex(0.630, 0.920, "#sqrt{s} = 8 TeV, L_{int} = 11.6 fb^{-1}")
   print()
   
   # Print cut-flow report.
   print( _GREEN + " >>> Print cut-flow report." +
          _RESET )
   report.Print()
   print()
   


if __name__ == "__main__":
   ntpl004_dimuon()
