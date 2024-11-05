## \file
## \ingroup tutorial_v7
##
## \macro_code
##
## \date 2019-05-29
## \warning 
##          This is part of the ROOT 7 prototype! 
##          It will change without notice. It might trigger earthquakes. 
##          Feedback is welcome!
##
## \authors Bertrand Bellenot <Bertrand.Bellenot@cern.ch>, Sergey Linev <S.Linev@gsi.de>
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
from ROOT import Experimental
from ROOT.Experimental import (
                                RCanvas,
                                )


# classes
from ROOT import (
                   RBrowser,
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
                   gStyle,
                   gPad,
                   gRandom,
                   gBenchmark,
                   gROOT,
                   gInterpreter,
                   )



try:
   gSystem.Load( "libROOTBrowserv7.so" )
except:
   raise RuntimeError(
         "libROOTBrowserv7.so not found",
         )

# C++ wrappers.
#
# "ClearOnClose" method of "RBrowser" class doesn't work 
# appropriately in PyRoot.
# We fix this issue by using cppyy wrapper method.
# Correspondance of method with function:
#        RBrowser::ClearOnClose     ->   ClearOnClose_Py 
import subprocess
command = ( " g++                              "                            
            "     -shared                      " 
            "     -fPIC                        " 
            "     -o browser_wrapper.so        " 
            "     browser_wrapper.cpp          " 
            "     `root-config --cflags --libs`" 
            "     -fmax-errors=1"               ,               
            )
# Then execute.
try : 
   subprocess.run( command, shell=True )
except :
   raise RuntimeError( "browser_wrapper.cpp function not well compiled.")

# Then load `browser_wrapper.cpp` .
cppyy.load_library( "./browser_wrapper.so" )
cppyy.include     ( "./browser_wrapper.cpp" )
from cppyy.gbl import (
                        ClearOnClose_Py,
                        )
# At last, the wrapper function is ready.



# void
def browser() :

   # Create browser.
   global br
   br = make_shared[ RBrowser ]() # RBrowser *
   
   # Clear when connection to client is being closed.
   # br.ClearOnClose( )     # Error.
   ClearOnClose_Py( br )    # Ok.
   


if __name__ == "__main__":
   browser()
