## \file
## \ingroup tutorial_v7
##
## \macro_code
##
## \date 2024-07-09
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


import time
import threading

import ROOT
import cppyy


# standard library
from ROOT import std
from ROOT.std import (
                       generate_canonical,
                       mt19937,
                       future,
                       iostream,
                       # random, # not available
                       make_shared,
                       unique_ptr,
                       )



# root 7
# experimental classes
from ROOT import Experimental
from ROOT.Experimental import (
                                RHistConcurrentFiller,
                                RHistConcurrentFillManager,
                                RHist,
                                RH2D,
                                Hist,
                                )

#



# classes
from ROOT import (
                   TH1,
                   TF1,
                   TCanvas,
                   )




# double
def wasteCPUTime(gen : std.mt19937 ) :

   """
   Simulate number crunching through gen and ridiculous num bits.

   """
   return std.generate_canonical [ float, 100 ] ( gen ) + \
          std.generate_canonical [ float, 100 ] ( gen ) + \
          std.generate_canonical [ float, 100 ] ( gen ) + \
          std.generate_canonical [ float, 100 ] ( gen ) + \
          std.generate_canonical [ float, 100 ] ( gen )
   


Filler_t = RHistConcurrentFiller[ RH2D, 1024 ]

# This function is called within each thread: it spends some CPU time and then
# fills a number into the histogram, through the Filler_t. This is repeated
# several times.
#
# void
def theTask(filler : Filler_t) :
   gen = std.mt19937()
   
   #for (int i = 0; i < 3000000; ++i) {
   #for i in range( 0, 3000000, 1) :  # Too large. Not tested.
   #for i in range( 0, 300000, 1) :   # Ok. large. Tested.
   for i in range( 0, 30000, 1) :    # Ok. 
   #for i in range( 0, 3000, 1) :     # Ok. 
   #for i in range( 0, 300, 1) :      # Ok.
   #for i in range( 0, 3, 1) :        # Ok # Fast Debug. 
      time_list = [ wasteCPUTime( gen ), wasteCPUTime( gen ) ] 
      coord_array_2d = Hist.RCoordArray[ 2 ]( time_list ) 
      filler.Fill( coord_array_2d )
   


# This example fills a histogram concurrently, from several threads.
#
# void
def concurrentHistFill( hist : RH2D) :

   """
   RHistConcurrentFillManager allows multiple threads to fill the histogram
   concurrently.
   
   Details: 
   Each thread's Fill() calls are buffered. Once the buffer is full,
   the "RHistConcurrentFillManager" locks and flushes the buffer 
   into the histogram.

   """

   global fillMgr
   fillMgr = RHistConcurrentFillManager[ RH2D ] ( hist )
   
   # Using Standard Template Library of cppyy "cppyy.std.thread".
   #
   # threads = std.array[ std.thread, 8 ]( ) 
   # 
   # # Let the threads fill the histogram concurrently.
   # #for thr in threads :
   #    # Each thread calls fill(), passing a dedicated filler per thread.
   #    # thr.__assign__( std.thread( theTask, fillMgr.MakeFiller() ) )
   #    # However, the above is not pythonized yet.
   #
   # # Join them.
   # for thr in threads: 
   #    thr.join


   # Using "threading" module of python instead.
   # 
   # Create the 8 threads.
   global threads
   threads = []

   for _ in range( 8 ):
      filler = fillMgr.MakeFiller()
      thr = threading.Thread( target= theTask, args=( filler, ) )
      # Appending a reference to the "thread" into the list "threads".
      threads.append( thr )
      thr.start()
   
   for thr in threads:
      thr.join()
      # print( "thr", thr ) 

   

# void
def concurrentfill() :

   """
   This histogram will be filled from several threads.

   """

   global hist
   hist = RH2D( [100, 0., 1. ], [0., 1., 2., 3., 10. ] )
   
   concurrentHistFill( hist )
   
   print( "threads * entries = ", hist.GetEntries() )
  
   hist.__destruct__()


if __name__ == "__main__":
   concurrentfill()
