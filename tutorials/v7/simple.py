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


import numpy as np

import cppyy
import ROOT


# standard library
from ROOT import std
from ROOT.std import (
                       make_shared,
                       unique_ptr,
                       )


# RooFit
cppyy.include( "ROOT/RFit.hxx" )
cppyy.include( "ROOT/RHistData.hxx" )
# "RFunction" only exists in namespace ROOT.Experimental
# if and only if "ROOT/RFit.hxx" has been loaded first.

# root 7
# experimental classes
from ROOT.Experimental import (
                                Hist,
                                RH2D,
                                RFunction, # Requires ROOT/RFit.hxx .
                                FitTo,     # Requires ROOT/RFit.hxx .
                                RFile,
                                # RFit,    # not available
                                RCanvas,
                                RHist,
                                RAxisConfig,
                                )


# classes
from ROOT import (
                   TH1D,
                   TCanvas,
                   TFile,
                   TH1F,
                   TGraph,
                   TLatex,
                   )

# maths
from ROOT import (
                   sin,
                   cos,
                   sqrt,
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


# c-integration
# Only works in ROOT 7.
# from ROOT.gInterpreter import (
#                                 ProcessLine,
#                                 Declare,
#                                 )
ProcessLine = gInterpreter.ProcessLine
Declare     = gInterpreter.Declare

# RooFit features.
ProcessLine('#include "ROOT/RFit.hxx"')
ProcessLine('#include "ROOT/RHistData.hxx"')
ProcessLine('using namespace ROOT::Experimental;')



# C++ wrappers.
#
# "FitTo" and "RFunction" functions and 
# "Draw" method of "RCanvas" class don't work appropriately in PyRoot.
# We fix this issue by using cppyy wrapper method.
# Correspondance of methods with functions:
#        RCanvas::Draw    ->    Draw_RH2D
#        FitTo            ->    FitTo_Py
#        RFunction<2>     ->    RFunction_2_Py
import subprocess
command = ( " g++                              "                            
            "     -std=c++20                   "  # module span requires c++20
            "     -shared                      " 
            "     -fPIC                        " 
            "     -o simple_wrapper.so         " 
            "     simple_wrapper.cpp           " 
            "     `root-config --cflags --libs`" 
            "     -fmax-errors=1"               ,               
            )
# Then execute.
try : 
   subprocess.run( command, shell=True )
except :
   raise RuntimeError( "simple_wrapper.cpp function not well compiled.")

# Then load `simple_wrapper.cpp` .
cppyy.load_library( "./simple_wrapper.so" )
cppyy.include     ( "./simple_wrapper.cpp" )
from cppyy.gbl import (
                        FitTo_Py,
                        RFunction_2_Py,
                        Draw_RH2D,
                        )
# At last, the wrapper function is ready.





# void
def simple() :
   
   # Create a 2D histogram with an X axis with equidistant bins, and a y axis
   # with irregular binning.
   xAxis = RAxisConfig( list( np.linspace(0., 1., num=100+1 ) ) )
   yAxis = RAxisConfig( [ 0., 1., 2., 3., 10. ] )
   histFromVars = RH2D(xAxis, yAxis)
   
   # Or the short in-place version:
   # Create a 2D histogram with an X axis with equidistant bins, and a y axis
   # with irregular binning.
   global hist
   hist = RH2D( 
                list( np.linspace(0., 1., num=100+1 ) ),  # x axis. Regular.
                [0., 1., 2., 3., 10.],                    # y axis. Iregular.
                )
   
   # Fill weight 1. at the coordinate 0.01, 1.02.
   global coord
   # coord = Hist.RCoordArray[2]( 0.01, 1.02 )  # Attention: It doesn't plot. 
   coord = Hist.RCoordArray[2]( 0.01, 1.00 )    # Ok.

   for i in range( 6 ):
      # TODO : This entry isn't being plot.
      # hist.Fill( Hist.RCoordArray[2]( 0.01, 1.02 ) )
      #
      hist.Fill( coord ) # Ok. At (0.01, 1.00) .
      # Note:   
      #       The reason why the coordinate isn't being plot
      #       differes from the C++ version. In its C++ version,
      #       we do not use np.linspace but the well implemented
      #       constructor RAxisConfig.
      #       Thus, the 2dim-mesh created is different along with
      #       its topological properties: closed or open.
      #                 (0.01, 1.02) open        
      #                 [0.01, 1.02) left-open
      #                 (0.01, 1.02] right-open
      #                 [0.01, 1.02] closed
      #       Creating regular sized bins have different properties
      #       than creating irregular sized, both in C++.
      #       In this PyRoot implementation, at using np.linspace 
      #       we mix even more the properties.
      #       That's why:
      #       >>> hist.Fill( Hist.RCoordArray[2]( 0.01, 1.02 ) )
      #       isn't show on the RCanvas. Besides of having being 
      #       filled previously.
      #       Still, more research needs to be done. TODO
      #       It fills, but plots nothing:
      #       >>> coord = Hist.RCoordArray[2]( 0.01, 1.02 )
      #       >>> hist.Fill( coord )  
      #       >>> hist.Fill( coord )  
      #       >>> hist.GetBinContent( coord ) # output: 2 # the two fills.
      #       2  

   for _ in range( 4 ):
      # Ok: This entry is being plot.
      hist.Fill( Hist.RCoordArray[2]( 0.51, 1.52 ) )  

   for _ in range( 5 ): 
      # Ok: This entry is being plot.
      hist.Fill( Hist.RCoordArray[2]( 0.31, 1.33 ) )  

   
   # Fit the histogram.
   # a.
   # def __( x: std.array[float,2], par: std.span[float] ) :
   def __( x: list, par: list ) : 
      # Polynomial:
      # P(x, y; a, b) = a x^2  +  b x  -  y^2
      return   par[0] * x[0] * x[0]    +  \
               par[1] * x[1]           +  \
               - x[1] * x[1]

   # For exploring func and fitResult
   # in the interpreter IPython[]: .
   global func, fitResult

   # b.
   # func = RFunction[2]( __ )                    # Error. 
   func = RFunction_2_Py( __ ) # * RFunction[2]   # Ok. 

   # c. 
   # fitResult = FitTo[""]( hist, func, [ 0., 1. ] )    # auto       # Error.
   fitResult = FitTo_Py( hist, func, [ 0., 1. ] )       # RFitResult # Ok.
   # Python style. Uncomment lines in wrapper C function.
   # fitResult, hist = FitTo_Py( hist, func, [ 0., 1. ] ) # std.pair 


   # Checking fit.
   # TODO.


   # Checking changes on a plot.
   global hist_p
   hist_p = std.shared_ptr[ RH2D ]( hist )
   
   global canvas
   canvas = RCanvas.Create( "RCanvas with histogram" )
   # canvas.Draw( hist_p )                 # error
   canvas = Draw_RH2D( canvas, hist_p )    # ok 
   canvas.Show()
   canvas.Update() 

   # Save the histogram.
   file = RFile.Recreate( "hist.root" )   # RFilePtr  # auto
   file.Write( "TheHist", hist )
   # TODO : Need to close file. Not done by Python.
   # file.Close() # Implement similar argument or improve pythonization.
   



if __name__ == "__main__":
   simple()
