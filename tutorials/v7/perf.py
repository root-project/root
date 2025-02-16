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
import cppyy


# standard library
from ROOT import std
from ROOT.std import (
                       span,
                       chrono,
                       iostream,
                       )




# root 7
# experimental classes
from ROOT import Experimental
from ROOT.Experimental import (
                                # RFit, # not available
                                RHist,
                                RHistBufferedFill,
                                RH2D,
                                Hist,
                                )

# classes
from ROOT import (
                   TFile,
                   TCanvas,
                   TH1,
                   TF1,
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


# timer
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
tmp_coord        = Hist.RCoordArray[ 2 ] ( [ 0.611, 0.611 ] )
tmp_HistBuffered.Fill( tmp_coord )
# Same for RH2D.FillN with std.span.
tmp_RH2D         = RH2D( [100, 0., 1.], [0., 1., 2., 3., 10.] )
tmp_v            = std.vector[ Hist.RCoordArray[ 2 ] ] ( 2 )
tmp_v.data()[0]  = 0.611  
tmp_v.data()[1]  = 0.611  
tmp_v_span       = span[ Hist.RCoordArray[ 2 ] ]( tmp_v.data(), tmp_v.size() )
tmp_RH2D.FillN( tmp_v_span )

# Clean up.
tmp_HistBuffered.__destruct__()
tmp_coord.__destruct__()
tmp.__destruct__()
del tmp, tmp_coord, tmp_HistBuffered, 
tmp_RH2D.__destruct__()
tmp_v.__destruct__()
tmp_v_span.__destruct__()
del tmp_RH2D, tmp_v, tmp_v_span
# Now some ROOT 7 classes(relevant for this tutorials) are loaded 
# in this PyRoot session.



# long
def createNew(count : int) :
   ret = 1; # long
   #for (int i = 0; i < count; ++i) {
   for i in range(0, count, 1):
      hist = RH2D( [100, 0., 1.], [0., 1., 2., 3., 10.] )
      ret ^= id( hist ) # long

      hist.__destruct__()
      
   return ret
   

# long
def fillNew(count : int) :
   hist = RH2D( [100, 0., 1.], [0., 1., 2., 3., 10.] )
   # for (int i = 0; i < count; ++i)
   for i in range( 0, count, 1) :
      hist.Fill( Hist.RCoordArray[2]( [ 0.611, 0.611 ] ) )

   hist_NDim = hist.GetNDim()
   # Clean up.
   hist.__destruct__()

   return hist_NDim 
   

# long
def fillN(count : int) :
   
   hist = RH2D( [100, 0., 1.], [0., 1., 2., 3., 10.] )
   v = std.vector[ Hist.RCoordArray[ 2 ] ] ( count )
   # for (int i = 0; i < count; ++i)
   for i in range( 0, count, 1) :
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

   # Clean up.
   v.__destruct__()
   hist.__destruct__()

   return hist_NDim 
   

# long
def fillBufferedNew(count : int) :
   hist = RH2D( [100, 0., 1.], [0., 1., 2., 3., 10.] )
   filler = RHistBufferedFill[ RH2D ] ( hist )
   # for (int i = 0; i < count; ++i)
   for i in range( 0, count, 1) :
      filler.Fill( Hist.RCoordArray[2]( [ 0.611, 0.611 ] ) )

   hist_NDim = hist.GetNDim()

   filler.__destruct__()
   hist.__destruct__()

   return hist_NDim 
   



from typing import Callable
timefunc_t = Callable[ [ int, ], int ]

# void
def time1(run : timefunc_t, count : int, name : str) :

   start = chrono.high_resolution_clock.now() # auto
   run( count )
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
def time(r7 : timefunc_t, count : int, name : str) :
   time1( r7, count, name + " (ROOT7)" )
   

# void
def perf() :

   # Too large.
   # time( fillN           , 1000000   , "fillN"               ) 
   # time( createNew       , 1000000   , "create 2D hists"     )
   # time( fillNew         , 100000000 , "2D fills"            )
   # time( fillBufferedNew , 100000000 , "2D fills (buffered)" )
   #
   # Test: Ok.


   print( " Starting test performance" ) 
   time( fillN           , 1000   , "fillN"               ) 
   time( createNew       , 1000   , "create 2D hists"     )
   time( fillNew         , 100000 , "2D fills"            )
   time( fillBufferedNew , 100000 , "2D fills (buffered)" )
   


if __name__ == "__main__":
   perf()
