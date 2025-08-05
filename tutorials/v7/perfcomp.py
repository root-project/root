## \file
## \ingroup tutorial_v7
##
## \macro_code
##
## \date 2024-07-08
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
import ctypes
from array import array


import gc
import ctypes

import ROOT
import cppyy


# standard library
from ROOT import std
from ROOT.std import (
                       span,
                       chrono,
                       iostream,
                       # type_traits, # not available
                       )


# root 7
# experimental classes
from ROOT import Experimental
from ROOT.Experimental import (
                                RH2D,
                                Hist,
                                # RFit, # not available
                                RHist,
                                RHistBufferedFill,
                                )


# classes
from ROOT import (
                   TH1,
                   TH2,
                   TH2D,
                   )


# globals
from ROOT import (
                   gROOT,
                   gDirectory,
                   gInterpreter,
                   )


# only root 7
# c-integration
# from ROOT.gInterpreter import (
#                                 ProcessLine,
#                                 Declare,
#                                 )
# all PyROOT versions
ProcessLine = gInterpreter.ProcessLine
Declare     = gInterpreter.Declare


Clock = std.chrono.high_resolution_clock





# Constructing the RH2D object for the first time in a PyRoot session
# could take time. And it is not part of the timing analysis. 
# So, let's construct and delete an RH2D so that PyRoot can load the 
# inner libraries.
tmp = RH2D (
              [ 0. , 0.1 , 0.3 , 1.        ] ,
              [ 0. , 1.  , 2.  , 3.  , 10. ] ,
              )
# Same for RHistBufferedFill and Hist.RCoordArray classes.
tmp_HistBuffered = RHistBufferedFill[ RH2D ] ( tmp )
tmp_coord = Hist.RCoordArray[ 2 ] ( [ 0.611, 0.611 ] )
tmp_HistBuffered.Fill( tmp_coord )
# Same for RH2D.FillN with std.span.
tmp_RH2D         = RH2D( [100, 0., 1.], [0., 1., 2., 3., 10.] )
tmp_v            = std.vector[ Hist.RCoordArray[ 2 ] ] ( 2 )
tmp_v.data()[0]  = 0.611  
tmp_v.data()[1]  = 0.611  
tmp_v_span       = span[ Hist.RCoordArray[ 2 ] ]( tmp_v.data(), tmp_v.size() )
tmp_RH2D.FillN( tmp_v_span )

tmp_HistBuffered.__destruct__()
tmp_coord.__destruct__()
tmp.__destruct__()
del tmp, tmp_coord, tmp_HistBuffered
# Now some ROOT 7 classes(relevant for this tutorials) are loaded 
# in this PyRoot session.




# long
def createNewII(count : int) :

   ret = 1 # long

   #for (int i = 0; i < count; ++i) {
   for i in range(0, count, 1):
      hist = RH2D (
                    [ 0. , 0.1 , 0.3 , 1.        ] ,
                    [ 0. , 1.  , 2.  , 3.  , 10. ] ,
                    )
      ret ^= id( hist ) # long
      hist.__destruct__()

   return ret
   


nBinsX, nBinsY = None, None 
x, y           = None, None
def BINSOLD():

   global nBinsX, nBinsY
   global x, y

   nBinsX = 4           
   nBinsY = 5           

   x = array( "d" , [ 0. , 0.1 , 0.3 , 1.       ] )
   y = array( "d" , [ 0. , 1.  , 2.  , 3. , 10. ] )


hist = None
def DECLOLD():

   global hist
   global nBinsX, nBinsY
   global x, y

   if not hist :
      hist = TH2D( "a", "a hist", nBinsX - 1, x, nBinsY - 1, y )
   else :
      gDirectory.Delete( "a" ) 
      hist = TH2D( "a", "a hist", nBinsX - 1, x, nBinsY - 1, y )


def OLD():

   global hist
   global nBinsX, nBinsY
   global x, y
   BINSOLD() # BINS OLD.
   DECLOLD() # OLD DECLaration.



# long
def createOldII(count : int) :

   global hist
   BINSOLD()
   ret = 1 # long

   #for (int i = 0; i < count; ++i) {
   for i in range(0, count, 1):
      DECLOLD()
      ret ^= id( hist ) # long

   # Retreiving Python object with: 
   # ctypes.cast( id(hist), ctypes.py_object).value

   gROOT.Remove( hist )
   gDirectory.Delete( "hist" )

   return ret
   

# long
def fillNewII(count : int) :

   hist = RH2D( [0., 0.1, 0.3, 1.], [0., 1., 2., 3., 10.] )

   #for (int i = 0; i < count; ++i)
   for i in range( 0, count, 1 ):
      # Not to use:
      # hist.Fill( [0.611, 0.611] )
      # Instead: 
      data_list = [ 0.611, 0.611 ]
      # Coordinate data for Histograms:
      coord_data = Hist.RCoordArray[ 2 ] ( data_list ) 
      hist.Fill( coord_data )

   hist.__destruct__()

   return hist.GetNDim()
   

# long
def fillOldII(count : int) :

   global hist
   OLD()

   #for (int i = 0; i < count; ++i)
   for i in range( 0, count, 1 ):
      # Not to use:
      # hist.Fill( [0.611, 0.611] )
      # Instead:
      hist.Fill( 0.611, 0.611 )

   gROOT.Remove( hist )

   return int( hist.GetEntries() ) # long
   

# long
def fillNII(count : int) :

   hist = RH2D( [ 0., 0.1, 0.3, 1. ], [ 0., 1., 2., 3., 10. ] )
   v = std.vector[ Hist.RCoordArray[ 2 ] ]  ( count )

   #for (int i = 0; i < count; ++i)
   for i in range( 0, count, 1 ):
      # Not to use:
      # v[i] = [ 0.611, 0.611 ]
      # Instead:
      v[i].data() [0] = 0.611
      v[i].data() [1] = 0.611

   # Create a span from the vector and pass it to FillN.
   # From C++:
   # hist.FillN(std::span<const Experimental::Hist::RCoordArray<2>>(v.data(), v.size()));
   # 
   # To Python :
   v_span = span[ Hist.RCoordArray[ 2 ] ]( v.data(), v.size() ) 
   hist.FillN( v_span ) 

   hist_NDim = hist.GetNDim()

   hist.__destruct__()
   v.__destruct__()
   del v

   return hist_NDim
   

# long
def fillBufferedOldII(count : int) :

   global hist
   OLD()
   hist.SetBuffer( TH1.GetDefaultBufferSize() )

   #for (int i = 0; i < count; ++i)
   for i in range( 0, count, 1 ):
      hist.Fill( 0.611, 0.611 )

   gROOT.Remove( hist )

   return int( hist.GetEntries() ) # long
   

# long
def fillBufferedNewII(count : int) :

   hist_ = RH2D( [0., 0.1, 0.3, 1.], [0., 1., 2., 3., 10.] )
   filler_ = RHistBufferedFill[ RH2D ] ( hist_ )

   #for (int i = 0; i < count; ++i)
   for i in range( 0, count, 1 ):
      # Not to use:
      # filler_.Fill( [ 0.611, 0.611 ] ) 
      # Instead: 
      data_list = [ 0.611, 0.611 ]
      # Coordinate data for Histograms:
      coord_data = Hist.RCoordArray[ 2 ] ( data_list ) 
      filler_.Fill( coord_data )

   # Clean up.
   filler_.__destruct__()
   del filler_
   gc.collect()

   return hist_.GetNDim()
   

# EQUIDISTANT

# long
def createNewEE(count : int) :

   ret = 1 # long

   #for (int i = 0; i < count; ++i) {
   for i in range(0, count, 1):
      hist = RH2D( [100, 0., 1.], [5, 0., 10.] )
      ret ^= id( hist ) # long
      hist.__destruct__()

   return ret
   

# long
def createOldEE(count : int) :

   ret = 1 # long

   #for (int i = 0; i < count; ++i) {
   for i in range(0, count, 1):
      hist = TH2D( "a", "a hist", 100, 0., 1., 5, 0., 10. )
      ret ^= id( hist ) # long
      gDirectory.Delete( "a" ) 

   return ret
   

# long
def fillNewEE(count : int) :

   hist = RH2D( [100, 0., 1.], [5, 0., 10.] )

   #for (int i = 0; i < count; ++i)
   for i in range( 0, count, 1 ):
      # Not to use:
      # hist.Fill( [0.611, 0.611] )
      # Instead: 
      data_list = [ 0.611, 0.611 ]
      # Coordinate data for Histograms:
      coord_data = Hist.RCoordArray[ 2 ] ( data_list ) 
      hist.Fill( coord_data )

   hist.__destruct__()

   return hist.GetNDim()
   

# long
def fillOldEE(count : int) :

   hist = TH2D("a", "a hist", 100, 0., 1., 5, 0., 10.)

   #for (int i = 0; i < count; ++i)
   for i in range( 0, count, 1 ):
      hist.Fill( 0.611, 0.611 )

   gROOT.Remove( hist )

   return int( hist.GetEntries() ) # long
   

# long
def fillNEE(count : int) :

   hist = RH2D( [100, 0., 1.], [5, 0., 10.] )
   v = std.vector[ Hist.RCoordArray[ 2 ] ] ( count )

   #for (int i = 0; i < count; ++i)
   for i in range( 0, count, 1 ):
      # Not to use:
      # v[i] = [ 0.611, 0.611 ]
      # Instead:
      v[i].data() [0] = 0.611
      v[i].data() [1] = 0.611


   # Create a span from the vector and pass it to FillN.
   # From C++:
   # hist.FillN(std::span<const Experimental::Hist::RCoordArray<2>>(v.data(), v.size()));
   # 
   # To Python :
   v_span = span[ Hist.RCoordArray[ 2 ] ]( v.data(), v.size() ) 
   hist.FillN( v_span ) 


   hist_NDim = hist.GetNDim()

   hist.__destruct__()
   v.__destruct__()
   del v

   return hist_NDim
   

# long
def fillBufferedOldEE(count : int) :

   hist = TH2D("a", "a hist", 100, 0., 1., 5, 0., 10.)
   hist.SetBuffer( TH1.GetDefaultBufferSize() )

   #for (int i = 0; i < count; ++i)
   for i in range( 0, count, 1 ):

      hist.Fill( 0.611, 0.611 )

   hist_Entries = hist.GetEntries()

   hist.__destruct__()

   return int( hist_Entries ) # long
   

# long
def fillBufferedNewEE(count : int) :

   hist = RH2D( [100, 0., 1.], [5, 0., 10.] )
   filler = RHistBufferedFill[RH2D] ( hist )

   #for (int i = 0; i < count; ++i)
   for i in range( 0, count, 1 ):
      # Not to use:
      # filler.Fill( [ 0.611, 0.611 ] ) 
      # Instead: 
      data_list = [ 0.611, 0.611 ]
      # Coordinate data for Histograms:
      coord_data = Hist.RCoordArray[ 2 ] ( data_list ) 
      filler.Fill( coord_data )

   # Clean up.
   filler.__destruct__()
   del filler
   gc.collect()
   
   return hist.GetNDim()
   

# From C++ :
#
# using timefunc_t = std.add_pointer_t<long(int)>
#
# To Python :
#
from typing import Callable
timefunc_t = Callable[ [ int, ], int ]
# Note :
#         "timefunc_t" is type alias for a function.
#         It takes an [ int, ] and returns and int; like:
#         >>> def f( int, ) -> int : pass



# void
def time1(run : timefunc_t, count : int, name : str) :

   # 1.
   start = chrono.high_resolution_clock.now() # auto
   # 2.
   run( count )
   # 3.
   end   = chrono.high_resolution_clock.now() # auto


   time_span = chrono.duration_cast[ chrono.duration[ float ] ] ( end - start ) # duration[double]
   
   print( count             ,
          " * "             ,
          name              ,
          ": "              ,
          time_span.count() ,
          "seconds "        ,
          )
   

# void
def time(r6 : timefunc_t, r7 : timefunc_t, count : int, name : str) :

   time1( r6, count, name + " (ROOT6)")
   time1( r7, count, name + " (ROOT7)")
   

# void
def perfcomp() :

   # factor = 1000000 # No tested. Too much time. TODO
   # factor = 1 # For fast debug!
   # factor = 3 # For memory tests.
   factor = 100 # For a normal test.
   # factor = 1000 # For a large test.
   # factor = 10000 # For a long test.

   time ( createOldII       , createNewII       , factor       , "create 2D hists [II]"     )
   time ( createOldEE       , createNewEE       , factor       , "create 2D hists [EE]"     )
   time ( fillOldII         , fillNewII         , 100 * factor , "2D fills [II]"            )
   time ( fillOldEE         , fillNewEE         , 100 * factor , "2D fills [EE]"            )
   time ( fillBufferedOldII , fillBufferedNewII , 100 * factor , "2D fills (buffered) [II]" )
   time ( fillBufferedOldEE , fillBufferedNewEE , 100 * factor , "2D fills (buffered) [EE]" )
   


if __name__ == "__main__":
   perfcomp()
