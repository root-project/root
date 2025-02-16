## \file
## \ingroup tutorial_v7
##
## \macro_code
##
## \date 2019-11-01
## \warning 
##          This is part of the ROOT 7 prototype! 
##          It will change without notice. It might trigger earthquakes. 
##          Feedback is welcome!
##
## \author Sergey Linev <S.Linev@gsi.de>
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
import ctypes
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
                   RFileDialog,
                   TH1,
                   TCanvas,
                   TFile,
                   TH1F,
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



# Description:
#
# Show how RFileDialog can be used in sync and async modes.
# Normally, file dialogs will be used inside other widgets like "ui5" dialogs.
# Here, by default, this dialog starts in async mode - means macro immediately 
# returns to command line.
# To start OpenFile dialog in sync mode, call `filedialog(1)"` 
# after running `%run filedialog.py`.
# Once a file is selected, the pyroot function's execution will be stopped.


if gSystem.Load( "libROOTBrowserv7" ) :
   print( "libROOTBrowserv7.so successfully loaded." )
else :
   raise ImportError( "libROOTBrowserv7.so not found." )


# void
def filedialog(kind : int = 0) :

   fileName = "" 
   
   # Example of sync methods, blocks until name is selected.
   match kind :
      case 1:
         fileName = RFileDialog.OpenFile ( "OpenFile title"                ) # str
      case 2:
         fileName = RFileDialog.SaveAs   ( "SaveAs title"  , "newfile.xml" ) # str
      case 3:
         fileName = RFileDialog.NewFile  ( "NewFile title" , "test.txt"    ) # str
      
    
   if kind > 0  :
      print("Selected file: %s\n" % fileName )
      return
      
   
   global dialog
   dialog = std.make_shared[ RFileDialog ](
                                            ROOT.RFileDialog.kOpenFile,
                                            "OpenFile dialog in async mode"
                                            )  # auto
   
   dialog.SetNameFilters( 
                          [ 
                            "C++ files (*.cxx *.cpp *.c *.C)"  ,
                            "ROOT files (*.root)"              ,
                            "Image files (*.png *.jpg *.jpeg)" ,
                            "Text files (*.txt)"               ,
                            "Any files (*)"                    ,
                            ]
                          )
   
   # Preselection by default at opening web browser.
   dialog.SetSelectedFilter( "ROOT files" )
   
   # Use dialog capture to keep reference until file name is selected.
   def __( res : str ) :
      print( "The Selected file: %s\n" % res )
      # Clean up dialog.
      # dialog.reset() # Fix error on C++ version.


   # No need for lambda function here. 
   # dialog.SetCallback( lambda res : __( res ) )
   # Calling the python function should do it.
   dialog.SetCallback( __ )
      
   
   dialog.Show()

   # TODO
   # At exit:
   #    terminate called after throwing an instance of 'CPyCppyy::PyException'
   #    what():  python exception

   


if __name__ == "__main__":
   filedialog()
