## \file
## \ingroup tutorial_v7
##
## \macro_code
##
## \date 2024-03-22
## \warning 
##          This is part of the ROOT 7 prototype! 
##          It will change without notice. It might trigger earthquakes. 
##          Feedback is welcome!
##
## \author Axel Naumann <axel@cern.ch>
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
                                Hist,
                                RCanvas,
                                RFitPanel,
                                RHist,
                                RHistDrawable,
                                #
                                RAxisConfig,
                                RH1D,
                                RDrawable,
                                RColor,
                                RHistDrawable,
                                )


# classes
from ROOT import (
                   TH1D,
                   TCanvas,
                   TFile,
                   TH1F,
                   TGraph,
                   TLatex,
                   TPad,
                   )

# maths
from ROOT import (
                   sin,
                   cos,
                   sqrt,
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



# Draw, AddPanel, and ClearOnClose function methods of RCanvas 
# doesn't work appropiately in PyRoot. 
# We fix this issue by using cppyy wrapper method.
# Correspondance of methods with functions:
#     RCanvas.Draw            ->    Draw_RH1D
#     RCanvas.AddPanel        ->    AddPanel_RFitPanel
#     RCanvas.ClearOnClose    ->    ClearOnClose_RFitPanel
import subprocess
command = ( " g++                              "                            
            "     -shared                      " 
            "     -fPIC                        " 
            "     -o fitpanel_wrapper.so       " 
            "     fitpanel_wrapper.cpp         " 
            "     `root-config --cflags --libs`" 
            "     -fmax-errors=1"               ,               
            )
# Then execute.
try : 
   subprocess.run( command, shell=True )
except :
   raise RuntimeError( "fitpanel_wrapper.cpp function not well compiled.")
# Then load `histogram_wrapper.cpp` .
cppyy.load_library( "./fitpanel_wrapper.so" )
cppyy.include     ( "./fitpanel_wrapper.cpp" )
from cppyy.gbl import (
                        Draw_RH1D,
                        AddPanel_RFitPanel,
                        ClearOnClose_RFitPanel, 
                        )
# At last, the wrapper function is ready.




# void
def fitpanel() :
   
   global xaxis
   xaxis = RAxisConfig(10, 0., 10.)

   # Create the histogram.
   global pHist
   pHist = std.make_shared[ RH1D ]( xaxis )  # auto
   
   # Fill a few points.
   pHist.Fill( Hist.RCoordArray[ 1 ] ( 1 ) )
   pHist.Fill( Hist.RCoordArray[ 1 ] ( 2 ) )
   pHist.Fill( Hist.RCoordArray[ 1 ] ( 2 ) )
   pHist.Fill( Hist.RCoordArray[ 1 ] ( 3 ) )
   
   global canvas 
   canvas = RCanvas.Create("RCanvas with histogram") # auto
   # Not to use:
   # canvas.Draw( pHist )
   # Instead:
   canvas = Draw_RH1D( canvas, pHist )

   # canvas.SetLineColor( RColor.kRed )
   
   canvas.Show()
   canvas.Update() # need to ensure canvas is drawn
   
   global panel
   panel = std.make_shared[RFitPanel]( "FitPanel Title" ) # auto
   
   # TODO: How combine there methods together?
   # Here std.shread_ptr[] on both sides.
   
   panel.AssignCanvas    ( canvas )
   panel.AssignHistogram ( pHist  )
   
   # Not to use:
   # canvas.AddPanel( panel  
   # Instead:
   canvas = AddPanel_RFitPanel( canvas, panel )
   
   # Preserve panel alive until connection is closed.
   # Not to use:
   # canvas.ClearOnClose(panel)
   # Instead:
   canvas = ClearOnClose_RFitPanel( canvas, panel )
   


if __name__ == "__main__":
   fitpanel()
