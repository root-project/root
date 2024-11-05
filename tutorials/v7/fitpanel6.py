## \file
## \ingroup tutorial_v7
##
## \macro_code
##
## \date 2024-04-11
## \warning 
##          This is part of the ROOT 7 prototype! 
##          It will change without notice. It might trigger earthquakes. 
##          Feedback is welcome!
##
## \authors Sergey Linev <S.Linev@gsi.de>, Iliana Betsou <Iliana.Betsou@cern.ch>
## \translator P. P.

'''
*************************************************************************
* Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.               *
* All rights reserved.                                                  *
*                                                                       *
* For the licensing terms see $ROOTSYS/LICENSE.                         *
* For the list of contributors see $ROOTSYS/README/CREDITS.             *
*************************************************************************
'''


import ROOT
import cppyy


# standard library
from ROOT import std
from ROOT.std import (
                       make_shared,
                       unique_ptr,
                       )

# root 7
# experimental classes
from ROOT.Experimental import (
                                RFitPanel,
                                RCanvas,
                                RColor,
                                RHistDrawable,
                                )


# classes
from ROOT import (
                   TFile,
                   TH1,
                   TH1F,
                   TH1D,
                   TCanvas,
                   TFile,
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
                   gSystem,
                   gFile,
                   gStyle,
                   gPad,
                   gRandom,
                   gBenchmark,
                   gROOT,
                   gInterpreter,
                   )


# C++ wrappers.
#
# "ClearOnClose" method of "RFitPanel" class doesn't work 
# appropriately in PyRoot.
# We fix this issue by using cppyy wrapper method.
# Correspondance of method with function:
#        RFitPanel::ClearOnClose     ->   ClearOnClose_Py 
import subprocess
command = ( " g++                              "                            
            "     -shared                      " 
            "     -fPIC                        " 
            "     -o fitpanel6_wrapper.so      " 
            "     fitpanel6_wrapper.cpp        " 
            "     `root-config --cflags --libs`" 
            "     -fmax-errors=1"               ,               
            )
# Then execute.
try : 
   subprocess.run( command, shell=True )
except :
   raise RuntimeError( "fitpanel6_wrapper.cpp function not well compiled.")

# Then load `fitpanel6_wrapper.cpp` .
cppyy.load_library( "./fitpanel6_wrapper.so" )
cppyy.include     ( "./fitpanel6_wrapper.cpp" )
from cppyy.gbl import (
                        ClearOnClose_Py,
                        )
# At last, the wrapper function is ready.





# void
def fitpanel6() :
    
   tut_dir = gROOT.GetTutorialsDir() 
   global file
   file = TFile.Open( tut_dir + "/hsimple.root") # TODO Fix on .cxx version.

   if (gFile)  :
      gFile.Get("hpx")
      gFile.Get("hpxpy")
      gFile.Get("hprof")
      
   # Create panel.
   panel = make_shared[ RFitPanel ]( "FitPanel" ) # auto
   
   test = TH1F( "test", "This is test histogram", 100, -4, 4 ) # new
   test.FillRandom( "gaus", 10000 )
   
   panel.AssignHistogram( test )
   
   panel.Show()
   
   # panel.ClearOnClose( panel )  # Error.
   ClearOnClose_Py( panel )       # Ok.

   # Feel free to explore the RFitPanel of ROOT7.
   


if __name__ == "__main__":
   fitpanel6()
