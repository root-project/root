## \file
## \ingroup tutorial_tree
## \notebook -nodraw
##
##
## Benchmark ...
## comparing the performance of 
## a row-wise and a column-wise storage 
## process.
##
## This test consists in writing/reading to/from keys or trees.
##
##
## To execute this benchmark, do:
## ~~~
## IPython [0]: %run bill.py
## ~~~
##
## For example, for N=10000 events, the following output is produced
## on an 2.7 GHz Intel Core i7 (year 2011).
##
## ~~~
## billw0  : RT=  0.803 s, Cpu=  0.800 s, File size=  45608143 bytes, CX= 1
## billr0  : RT=  0.388 s, Cpu=  0.390 s
## billtw0 : RT=  0.336 s, Cpu=  0.310 s, File size=  45266881 bytes, CX= 1.00034
## billtr0 : RT=  0.229 s, Cpu=  0.230 s
## billw1  : RT=  1.671 s, Cpu=  1.670 s, File size=  16760526 bytes, CX= 2.72078
## billr1  : RT=  0.667 s, Cpu=  0.680 s
## billtw1 : RT=  0.775 s, Cpu=  0.770 s, File size=   9540884 bytes, CX= 4.74501
## billtr1 : RT=  0.352 s, Cpu=  0.350 s
## billtot : RT=  5.384 s, Cpu=  5.290 s
## ******************************************************************
## *  ROOTMARKS =1763.9   *  Root6.05/03   20150914/948
## ******************************************************************
##
## In here, the first column categorizes the different processes
## featuring reading and writing ( letters "r" and "w") 
## with or without involving a Tree ( letter "t" ).
## Wherein also, the 'bill' prefix name refers to
## a "billing time" for processing time or task duration.
## On the second and third column are the timing (relative time and cpu time)
## to complete theirs tasks. Litle values signify fast execution, and higher 
## values slow execution, relatively.
## So, the faster the performance, the better.
##
## 
## ~~~
## \macro_code
## \author Rene Brun
## \translator P. P.


import ROOT
import ctypes
from array import array



# standard library
from ROOT import std
from ROOT.std import (
                       make_shared,
                       unique_ptr,
                       )

# classes
from ROOT import (
                   TFile,
                   TH1,
                   TKey,
                   TROOT,
                   TRandom,
                   TStopwatch,
                   TSystem,
                   TTree,
                   TIter,
                   TCanvas,
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

# types
from ROOT import (
                   Double_t,
                   Bool_t,
                   Float_t,
                   Int_t,
                   nullptr,
                   char,
)
#
from ctypes import c_double

# utils
def to_c( ls ):
   return (c_double * len(ls) )( * ls )
def printf(string, *args):
   print( string % args, end="")
def sprintf(buffer, string, *args):
   buffer = string % args 
   return buffer

# constants
from ROOT import (
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
)



# Number of events to be processed.
#N = 0 # Int_t  # ok
#N = 10 # Int_t # ok
N = 1000 # Int_t # ok
#N = 10000 # Int_t # error
timer = TStopwatch()

# void
def billw(billname : char, compress : Int_t) :

   """
   Billing time or processing time for READING 
   histograms N=10000 histograms on a .root file.
   """

   timer.Start()

   #
   # Write N histograms as keys in a Tree in a File automatically.
   global f, h
   f = TFile(billname, "recreate", "bill benchmark with keys", compress)
   h = TH1F("h", "h", 1000, -3, 3)
   h.FillRandom("gaus", 50000)
   # 
   #for (Int_t i = 0; i < N; i++) {
   for i in range(0, N, 1):
      #
      # inherited from C ... 
      # name = " "*20
      # name = sprintf(name, "h%d", i)
      # pythonic... 
      name = "h%d" % i # pythonic... 
      #
      # Modifications on i-th histogram.
      h.SetName(name)
      h.Fill( 2 * gRandom.Rndm() ) 
      #
      # Writing on file. 
      h.Write()
   
   
   

   #
   timer.Stop()


   #
   printf("billw%d  : RT=%7.3f s, Cpu=%7.3f s, File size= %9d bytes, CX= %g\n",
          compress,
          timer.RealTime(),
          timer.CpuTime(),
          int( f.GetBytesWritten() ),
          f.GetCompressionFactor(),
          )

   #
   f.Close()
   #
   gROOT.Remove( f )
   del f

   ##
   del h


# void
def billr(billname : char, compress : Int_t) :

   """
   Billing time or processing time for READING 
   histograms N=10000 histograms from a .root file
   using keys. 
   We get all its keys from the .root file and
   then read its belonging object, the histogram.
   Furthermore we compute a statistical mean from
   each histogram, without no purpose but of 
   timing its process. So, no graphics involved. 
    
   
   Where N is a global variable.
   "billname" is the output.
   "compress" is the level of compression: 0,1,2,...,
   or 9; being 0 no compression at all, and 9 the 
   highest compression possible.

   But here "compress" doesn't have any effect at all.
   It is only a paramter for printing output.

   """


   #
   timer.Start()
  
   #
   # Read N histograms from keys.
   #

   #
   global f
   f = TFile( billname )
  
   # Iterables.
   #
   global h, key
   h   = TH1F()
   key = TKey()
   #
   # Efficient managament for N=10000 histograms
   # by taking them out from the current ROOT file structure.
   TH1.AddDirectory( False )

   #
   # Here itwll come the statistical means obtained from all 
   # histograms by reading the TKey objects( or keys ).  
   #
   global hmean
   hmean = TH1F("hmean", "hist mean from keys", 100, 0, 1)  # TH1F

   #
   # Iterator of keys.
   #
   global next_, list_of_keys
   list_of_keys = f.GetListOfKeys()           # THashList 
   next_        = TIter( f.GetListOfKeys() )  # TIter ( THashList )

   #
   # Loop.
   key   = next_  # (TKey *)
   i     = 0        # Int_t
   #
   while ( key:= next_() )  :
      # 
      if i > 100 : 
         Warning( "Too much! Possible memory leak.")
         break
      #
      h = key.ReadObj() # (TH1F *)
      hmean.Fill( h.GetMean() )
      #
      gROOT.Remove( h )
      del h
      #
      i += 1
   # 
   # # Alternatively, to avoid memory leaks use "for".
   # for key in list_of_keys :
   #    #
   #    print( "key:",i,",", end="" )
   #    #
   #    hist = key.ReadObj() # TH1F *
   #    #
   #    hmean.Fill( hist.GetMean(   ) )
   #    #
   #    gROOT.Remove( hist )
   #    del hist
   #    #
   #    i += 1
   # ... 
   #
   timer.Stop()


   #
   # Outout.
   printf("billr%d  : RT=%7.3f s, Cpu=%7.3f s\n",
          compress,
          timer.RealTime(),
          timer.CpuTime(),
          )
   


# void
def billtw(billtname : char, compress : Int_t) :

   """
   Billing time or processing time for WRITING on a TREE :
   N=10000 histograms generated on run time, one step ahead 
   each.
   The TREE is stored in a .root file using a user-defined 
   level of compression.
   
   Where N is a global variable.
   "billname" is the output.
   "compress" is the level of compression of the file: 
   0,1,2,..., or 9; being 0 no compression at all,
   and 9 the highest compression possible.

   """

   #
   timer.Start()

   #
   # Write N histograms into a Tree.
   #
   global f
   f = TFile(billtname, "recreate", "bill benchmark with trees", compress)
   #
   global h
   h = TH1F("h", "h", 1000, -3, 3)  # TH1F
   h.FillRandom("gaus", 50000)
   #
   #  
   global T
   T = TTree("T", "test bill")  # TTree
   T.Branch("event", "TH1F", h, 64000, 0)
   # Note: 64000 is the maximum number of TH1F objects
   #       to be stored. Our analysis requires only 10000
   #       objects. The performance was not tested over
   #       its maximum capacity over around 64000. 
   #       Memory issues could happen. Be aware.

   #
   # Filling N=10000 »sligthly« different histograms
   # into the tree.
   #
   #for (Int_t i = 0  i < N  i++) {
   for i in range(0, N, 1):
      #
      # i-th histogram. 
      global name
      name = "h%d" % i
      h.SetName( name )
      # The i-th histogram will have +i more entries 
      # than the one before.
      h.Fill(2 * gRandom.Rndm())
      #
      #
      T.Fill()
   # ...
   #
   T.Write()
   gROOT.Remove( T )
   del T


   #
   timer.Stop()

   #
   printf("billtw%d : RT=%7.3f s, Cpu=%7.3f s, File size= %9d bytes, CX= %g\n",
          compress,
          timer.RealTime(),
          timer.CpuTime(),
          int( f.GetBytesWritten() ),
          f.GetCompressionFactor(),
          )

   #
   f.Close()
   #
   gROOT.Remove( f )
   del f

   

# void
def billtr(billtname : char, compress : Int_t) :

   """
   Billing time or processing time for READING from a TREE :
   N=10000 histograms.
   The TREE has to be stored previously in a .root file.
   
   
   - Where N is a global variable.
   - "billname" is the output.
   - "compress" is the level of compression of the file: 
     0,1,2,..., or 9; being 0 no compression at all,
     and 9 the highest compression possible.
     But here "compress" doesn't have any effect at all.
     It is only a paramter for printing output.
     Tip: You can get the level of comression by simply 
          using TFile.GetLevelOfCompression().

   """

   #
   timer.Start()

   #
   # Read N histograms from a tree.
   #
   f = TFile( billtname )
   #
   T = f.Get("T")  # (TTree *)
   #
   h = TH1F()      # TH1F * # nullptr 
   T.SetBranchAddress("event", h)
   #
   #
   hmeant = TH1F("hmeant", "hist mean from tree", 100, 0, 1)  # TH1F
   #
   # Loop.
   nentries = T.GetEntries()  # Long64_t
   #
   #for (Long64_t i = 0  i < nentries  i++) {
   for i in range(0, nentries, 1):
      #
      T.GetEntry( i )
      #
      # "h" is connected to "T". 
      #If "T" changes, so does "h". 
      hmeant.Fill( h.GetMean() )


   #
   # The "T" tree has just been read
   # by collecting the statistical means, of each 
   # histogram stored in "T", in "hmeant".
      
   #
   timer.Stop()

   #
   printf("billtr%d : RT=%7.3f s, Cpu=%7.3f s\n",
          compress,
          timer.RealTime(),
          timer.CpuTime(),
          )
   

# void
def bill() :

   """
   Main function for the test.
 
   """

   #
   # A bit redundant, but it'll work for GitHub automatic launches.
   try:
      __FILE__ = __file__
   except:
      __FILE__ = "bill.py"

   #
   Dir = gSystem.GetDirName( gSystem.UnixPathName( __FILE__ ) ) # TString
   Dir = str( Dir ) 
   bill = Dir + "/bill.root"    # TString
   billt = Dir + "/billt.root"  # TString
   
   #
   totaltimer = TStopwatch()
   totaltimer.Start()

   #
   # Testing Levels of Compression : 0, 1.
   # 
   #for (Int_t compress = 0  compress < 2  compress++) {
   for compress in range(0, 2, 1):
      #
      billw(bill, compress)
      billr(bill, compress)
      billtw(billt, compress) # error
      billtr(billt, compress)
      
   #
   # Delete objects from ROOT.
   gSystem.Unlink ( bill  )
   gSystem.Unlink ( billt )
   

   #
   totaltimer.Stop()
   #
   realtime = totaltimer.RealTime()  # Double_t
   cputime  = totaltimer.CpuTime()   # Double_t

   #
   printf("billtot : RT=%7.3f s, Cpu=%7.3f s\n", realtime, cputime)




   #
   # The reference time is a Pentium-IV 2.4 GHz
   rootmarks = 600 * (16.98 + 14.40) / (realtime + cputime)  # Float_t
   #
   printf("******************************************************************\n")
   printf("*  ROOTMARKS =%6.1f   *  Root%-8s  %d/%d\n",
          rootmarks,
          gROOT.GetVersion(),
          gROOT.GetVersionDate(),
          gROOT.GetVersionTime(),
          )
   printf("******************************************************************\n")
   


if __name__ == "__main__":
   bill()
