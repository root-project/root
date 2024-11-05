## \file
## \ingroup tutorial_v7
##
## \macro_code
##
## \date 2024-08-08
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
from array import array
import ctypes

import ROOT
import cppyy


# standard library
from ROOT import std
from ROOT.std import (
                       make_shared,
                       unique_ptr,
                       iostream,
                       )



# root 7
# experimental classes
from ROOT import Experimental
from ROOT.Experimental import (
                                Add,
                                Hist,
                                RAxisConfig,
                                RHist,
                                RH2D,
                                )




# classes
from ROOT import (
                   TCanvas,
                   TFile,
                   TH1F,
                   TGraph,
                   TCanvas,
                   )




# "Add" global function for RH2D objects, and GetBinContent and GetBinIndex 
# global methods of RH2D class doesn't work appropriately in PyRoot.
# We fix this issue by using cppyy wrapper method.
# Correspondance of methods with functions:
#     Add                               ->    Add_RHist
#     RH2D::GetImpl()->GetBinIndex      ->    GetBinIndex
#     RH2D::GetImpl()->GetBinContent    ->    GetBinContent
import subprocess
command = ( " g++                              "                            
            "     -shared                      " 
            "     -fPIC                        " 
            "     -o histops_wrapper.so        " 
            "     histops_wrapper.cpp          " 
            "     `root-config --cflags --libs`" 
            "     -fmax-errors=1"               ,               
            )
# Then execute.
try : 
   subprocess.run( command, shell=True )
except :
   raise RuntimeError( "histops_wrapper.cpp function not well compiled.")
# Then load `histogram_wrapper.cpp` .
cppyy.load_library( "./histops_wrapper.so" )
cppyy.include     ( "./histops_wrapper.cpp" )
from cppyy.gbl import (
                        Add_RHist,
                        GetBinIndex,
                        GetBinContent,
                        )
# At last, the wrapper function is ready.




# void
def histops() :

   # Note:
   #       RAxisConfig is not fully pythonized in ROOT < 6.34.
   #       It only accepts one kind of initialization:
   #       |   Help on CPPOverload in module cppyy:
   #       |   __init__(...)
   #       |   ...
   #       |   RAxisConfig::RAxisConfig(vector<double>&& binborders)
   #       |   ...
   #       Which is the one that accepts a python-list as input parameter.
   #       Here the values of the list represent the limits of the bins.
   #       For example:
   #       >>> l = list( 0., 1., 2., 3., 10.0 ) 
   #       Will represent:
   #           Bin 1: [0., 1.)
   #           Bin 2: [1., 2.)
   #           Bin 3: [2., 3.)
   #           Bin 4: [3., 10.]
   #       This is useful for irregular bin sizes over an axis.
   #       However, for regular initialization of a large interval, 
   #       like 10 bins from 0. to 1., we will have to use Python
   #       features to create a list with 10 + 1 equally spaced 
   #       real numbers from 0. to 1.
   #       Like :
   #       >>> values = [i / 10 for i in range(11)]
   #       Or :
   #       >>> import numpy as np 
   #       >>> values = list( np.linspace( 0., 1., num=10+1 ) ) 
   #       Then initialize RAxisConfig with that list:
   #       >>> axis = RAxixConfig( values ) # values : list
   

   # - - - 1 - - -
   # Create a 2D histogram with an X axis with equidistant bins, and a y axis
   # with irregular binning.

   # Regular axis of bins.
   x_axis = RAxisConfig( list( np.linspace( 0., 1., num=10 + 1 ) ) ) 
   # Iregular axis of bins.
   y_axis = RAxisConfig( [0., 1., 2., 3., 10.] ) 

   global hist1
   # hist1 = RH2D( [100, 100., 100.], [ [0., 1., 2., 3., 10.] , ] )  # Bad.
   # hist1 = RH2D( [10, 0., 1.],      [0., 1., 2., 3., 10.]       )  # Bad.
   hist1 = RH2D( x_axis, y_axis )                                    # Correct.
   
   # Fill weight 1. at the coordinate 0.01, 1.02. .
   coordinates1 = Hist.RCoordArray[2]( 0.01, 1.02 ) 
   hist1.Fill( coordinates1 )      # Implicit expression.
   # hist1.Fill( coordinates1, 1 ) # Explicit expression.

   # Fill weight 0.5 at the coordinate 0.01, 1.02.
   # hist1.Fill( Hist.RCoordArray[2]( 0.01, 1.02 ), 0.5 )

   
   # - - - 2 - - -
   # Another 2D histogram with the same properties.

   # Regular axis of bins.
   x_axis = RAxisConfig( list( np.linspace(0., 1. , num=10+1 ) ) ) 
   # Iregular axis of bins.
   y_axis = RAxisConfig( [0., 1., 2., 3., 10.] ) 

   global hist2
   # hist2 = RH2D( [10, 0., 1.], [0., 1., 2., 3., 10.] ) # Bad.
   hist2 = RH2D( x_axis, y_axis )                        # Correct.


   # Fill weight 1. at the coordinate 0.02, 1.03 (that's the same bin).
   coordinates2 = Hist.RCoordArray[2]( 0.02, 1.03 ) 
   hist2.Fill( coordinates2 )      # Implicit expression.
   # hist2.Fill( coordinates2, 1 ) # Explicit expression.

   # Fill weight 0.5 at the coordinate 0.02, 1.03
   # hist2.Fill( Hist.RCoordArray[2]( [0.02, 1.03] ), 0.5 )
   

   # - - - 3 - - -
   # Addition of histograms.

   # Not to use:
   #Add( hist1, hist2 )
   # Instead:
   hist1 = Add_RHist( hist1, hist2 )
   


   # - - - 4 - - -
   # After adding histograms, obtain the result.
   global bin_coor, bin_idx, bin_cont
   bin_coor = Hist.RCoordArray[2]( 0.01, 1.02 )

   # Use Either:
   bin_idx  = hist1.GetImpl().GetBinIndex   ( bin_coor )         # int
   bin_cont = hist1.GetImpl().GetBinContent ( bin_coor  )        # float 

   # Or :
   # bin_idx  = GetBinIndex   ( hist1, bin_coor ) # int
   # bin_cont = GetBinContent ( hist1, bin_idx  ) # int

   print( "At bin_coor ", bin_coor )
   print( "bin_idx  ", bin_idx  )
   print( "bin_cont ", bin_cont )
   


if __name__ == "__main__":
   histops()
