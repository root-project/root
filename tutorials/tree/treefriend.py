## \file
## \ingroup tutorial_tree
## \notebook
##
##
## Illustrates how to use friend Trees:
##   - create a simple TTree
##   - Copy a "subset" of this TTree to a new TTree
##   - Create a Tree Index
##   - Make a friend TTree
##   - compare two TTrees, a normal with a friend.
##   - Draw a variable from the first Tree versus a variable
##     from the friend Tree
##
## You can run this tutorial with:
## ~~~
##  IPython [1] %run treefriend.py 
## ~~~
##
##
##
## \macro_output
## \macro_image
## \macro_code
##
## \author Rene Brun
## \translator P. P.


import ctypes
import numpy as np
from array import array

import ROOT



# standard library
from ROOT import std
from ROOT.std import (
                       make_shared,
                       unique_ptr,
                       )

# classes
from ROOT import (
                   TTree,
                   TFile,
                   TRandom,
                   TFile,
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
                   )
#
from ctypes import (
                   c_double,
                   c_float,
                   c_int,
)

#
#int32 = numpy.int32
#float32 = numpy.float32
# Note
#       Those are immutable types. 
#       Not useful for our purposes.
#       New redifinitions are given below.

#utils
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
                   gStyle,
                   gPad,
                   gRandom,
                   gBenchmark,
                   gROOT,
)


#
def float32(): return np.zeros( 1, dtype = "f" )
def int32():   return np.zeros( 1, dtype = "i" )
#
# TODO:
#       Item assigment to float32 or int32 is
#       done by using `[0]`. Improve a pythonization
#       by re-loading the `=` or `.value` or maybe
#       a method `.update()`.




# variables
Run, Event  = ( int32()  for _ in range( 2 ) )
x, y, z     = ( float32() for _ in range( 3 ) )
##
#fRun, fEvent = ( int32()  for _ in range( 2 ) )
#fx, fy, fz   = ( float32()for _ in range( 3 ) )
# 
# Note:
#       You can either define all the friend variables
#       outside of the scope of the function or inside.
#       Since we are trying to convey the meaning of
#       friend trees, friend branches, friend files,...,
#       we'll declare its definition in the scope of 
#       the their definition. It works either way.


# void
def CreateParentTree() :
   global Run, Event
   global x, y, z
   #
   # Create a simple TTree with 5 branches.
   # Two branches ("Run" and "Event") of it will be used to index the Tree.
   #
   global f, T
   f = TFile("treeparent.root", "recreate") # TFile
   T = TTree("T", "test friend trees")      # TTree
   #
   # 5 branches.
   T.Branch("Run", Run, "Run/I")
   T.Branch("Event", Event, "Event/I")
   T.Branch("x", x, "x/F")
   T.Branch("y", y, "y/F")
   T.Branch("z", z, "z/F")
   

   #
   global r
   r = TRandom()
   #
   #for (Int_t i = 0; i < 10000; i++) {
   for i in range(0, 10000, 1):
      if (i < 5000)  :
         #Run = int32( 1 )
         Run[0] = 1
         
      else  :
         Run[0] = 2
         
      #
      Event[0] = Float_t( i ) 
      #
      x[0]     = r.Gaus   ( 10, 1 )
      y[0]     = r.Gaus   ( 20, 2 )
      z[0]     = r.Landau (  2, 1 )
      #

      T.Fill()

   # 
   T.Print()
   T.Write()


   #
   f.Close()

   #
   #del f
   gROOT.Remove( f )
   del f
   
# void
def CreateFriendTree() :
   global Run, Event
   global x, y, z

   """
   Open the file created by CreateParentTree.
   Copy a subset of the TTree into a new TTree.
   Create an index on the new TTree ("Run","Event")
   Write the new TTree (including its index)
   
   Footnote:
            Please, see also tutorials copytree.C, copytree2.C and copytree3.C
            to understand better in deep how the "copy" process works in Trees.
   
   """

   
   #
   # The famous file and tree syntax as always.
   global f, T
   f = TFile("treeparent.root") # TFile
   T = f.Get("T") # (TTree *)

   #
   # Now.
   # The friend file. It is another simple file.
   global ff 
   ff = TFile("treefriend.root", "recreate") # TFile
   #
   # Then.
   # The friend tree, which needs a copied tree.
   global FT
   FT = T.CopyTree("z<10") # TTree *
   #
   # Proper setting-up for a friend tree with two indexes.
   FT.SetName("FT")
   FT.BuildIndex("Run", "Event")
   #
   # At last, we save it, and if we like, print it.
   FT.Write()
   FT.Print()
   

   #
   f.Close()
   ff.Close()


   #
   gROOT.Remove( ff )
   del ff

   #
   # Grammatical 
   # Note:
   #      "File friend" or "friend file",
   #      that is the question, Sir!.
   #      In the .C version, it is written "file friend"
   #      which is confusing concept for what we intend
   #      to convey: a file which is a friend of another file.
   #      "Au contraire de," a friend made of a file or a friend
   #      filed up, or a friend with files, ... 
   #      Here, we want to modify the noun "file" with
   #      another descriptive one "friend". At forming
   #      new concepts the description comes before the 
   #      characterization. 
   #      We'll stick with "friend file" from now on.
   #      
    

# void
def CompareTrees() :
   global Run, Event
   global x, y, z

   """
   The two TTrees created above are compared.
   The subset of entries in the friend TTree must be identical
   to the entries in the original TTree.
   Note: A friend tree is not necessarily a small 
         than the original. But it is frequently
         used in that way. 
         A huge tree with a small tree; where the 
         small one is the friend one.

   """
   
   global f, T
   f  = TFile("treeparent.root")  # TFile
   T  = f.Get("T")                # (TTree *)
   # 
   global ff, FT
   ff = TFile("treefriend.root")  # TFile
   FT = ff.Get("FT")              # (TTree *)
   #
   ##
   #fRun, fEvent = ( int32()  for _ in range( 2 ) )
   #fx, fy, fz   = ( float32()for _ in range( 3 ) )
   #
   global fRun, fEvent
   global fx, fy, fz
   fRun, fEvent = ( int32()   for _ in range( 2 ) )
   fx, fy, fz   = ( float32() for _ in range( 3 ) )
   

   #
   # Setting-up the Tree.
   #
   T.SetBranchAddress("Run", Run)
   T.SetBranchAddress("Event", Event)
   T.SetBranchAddress("x", x)
   T.SetBranchAddress("y", y)
   T.SetBranchAddress("z", z)
   #
   #
   # And Setting-up the other Tree.
   # Still not a friend.
   #
   FT.SetBranchAddress("Run", fRun)
   FT.SetBranchAddress("Event", fEvent)
   FT.SetBranchAddress("x", fx)
   FT.SetBranchAddress("y", fy)
   FT.SetBranchAddress("z", fz)
   #
   #
   # Adding the second tree to the first one
   # as a friend.
   T.AddFriend(FT) # Error
   #
   # Now. "FT" is a friend of "T".


   #   
   #
   nentries = T.GetEntries()  # Long64_t
   nok = 0  # Int_t
   #
   #for (Long64_t i = 0; i < nentries; i++) {
   for i in range(0, nentries, 1):
      #
      T.GetEntry(i)
      # Debug:
      #print( FT.GetEntryWithIndex( Run.item(), Event.item() ) )
      #print( "Run", Run, "fRun", fRun )
      #
      # Checking 
      if ( Run    == fRun   and \
           Event  == fEvent and \
           x      == fx     and \
           y      == fy     and \
           z      == fz         \
           )  :
         #
         nok += 1
         
      #
      else  :
         #
         # Not to use :
         # if (FT.GetEntryWithIndex(Run, Event) > 0)  :
         # Instead :
         # if (FT.GetEntryWithIndex( Int_t( Run ), Int_t( Event ) ) > 0)  :
         # Or Instead :
         if (FT.GetEntryWithIndex( Run.item(), Event.item() ) > 0)  :
         #
         #if (FT.GetEntryWithIndex(Run.value, Event.value) > 0)  :
            #
            if (i < 100)  :
               printf("i=%d, Run=%d, Event=%d, x=%g, y=%g, z=%g,"  +
                      " : fRun=%d, fEvent=%d, fx=%g, fy=%g, fz=%g\n",
                      i,
                         Run.item (),
                       Event.item (),
                           x.item (),
                           y.item (),
                           z.item (),
                        fRun.item (),
                      fEvent.item (),
                          fx.item (),
                          fy.item (),
                          fz.item (),
                      )
               
            
         
      
   #printf("nok = %d, fentries=%lld\n", nok, FT.GetEntries())
   printf("nok = %d, fentries=%d\n", nok, FT.GetEntries())


   #
   f.Close()
   ff.Close()
   
   gROOT.Remove( f )
   gROOT.Remove( ff )
   #del f
   #del ff
   

# void
def DrawFriend() :
   global Run, Event
   global x, y, z

   """
   Draw a scatter plot of the variable x in the parent tree versus
   the same variable in the friend tree.
   This should produce points along a straight line.
   
   """

   global f, T
   f = TFile.Open("treeparent.root") # TFile *
   T = f.Get("T"); # (TTree *)

   #
   # Adding a tree as a friend from root file 
   # with another tree directly.
   #
   T.AddFriend("FT", "treefriend.root")
   
   #
   T.Draw("x:FT.x")
   #
   # Note:
   #       The operator "." is stronger in 
   #       precendence than the ":". 
   #       So, "." will be evaulated first.
   

# void
def treefriend() :
   #
   global Run, Event
   global x, y, z
   #
   CreateParentTree()
   CreateFriendTree()
   CompareTrees()
   DrawFriend()
   


if __name__ == "__main__":
   treefriend()
