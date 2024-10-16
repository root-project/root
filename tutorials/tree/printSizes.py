## \file
## \ingroup tutorial_tree
## \notebook -nodraw
##
## This script can be used to get aggregate information
## of the size taken on the disk( or memory) by various
## differents branchs in a TTree object.
##
## For example:
##
## ~~~{.py}
##
## IPython [1]: %run printSizes.py
##
## IPython [2]: printTreeSummary(tree) # tree is global after %1.
##
##     Out [2]: The TTree "T" takes 3764343 bytes on disk.
##              Its branch "event" takes 3760313 bytes on disk.
##
## IPython [3]: printBranchSummary(tree.GetBranch("event")) 
##
##     Out [3]:
##              The branch "event" takes 3760313 bytes on disk.
##                Its sub-branch "TObject" takes 581 bytes on disk.
##                Its sub-branch "fType[20]" takes 640 bytes on disk.
##                Its sub-branch "fEventName" takes 855 bytes on disk.
##                Its sub-branch "fNtrack" takes 506 bytes on disk.
##                Its sub-branch "fNseg" takes 554 bytes on disk.
##                Its sub-branch "fNvertex" takes 507 bytes on disk.
##                Its sub-branch "fFlag" takes 420 bytes on disk.
##                Its sub-branch "fTemperature" takes 738 bytes on disk.
##                Its sub-branch "fMeasures[10]" takes 1856 bytes on disk.
##                Its sub-branch "fMatrix[4][4]" takes 4563 bytes on disk.
##                Its sub-branch "fClosestDistance" takes 2881 bytes on disk.
##                Its sub-branch "fEvtHdr" takes 847 bytes on disk.
##                Its sub-branch "fTracks" takes 3673982 bytes on disk.
##                Its sub-branch "fHighPt" takes 59640 bytes on disk.
##                Its sub-branch "fMuons" takes 1656 bytes on disk.
##                Its sub-branch "fLastTrack" takes 785 bytes on disk.
##                Its sub-branch "fWebHistogram" takes 596 bytes on disk.
##                Its sub-branch "fH" takes 10076 bytes on disk.
##                Its sub-branch "fTriggerBits" takes 1699 bytes on disk.
##                Its sub-branch "fIsValid" takes 366 bytes on disk.
## ~~~
##
## \macro_code
##
## \author
## \translator P. P.


import sys
import ctypes
from array import array
from functools import singledispatch

import ROOT


# standard library
from ROOT import std
from ROOT.std import (
                       make_shared,
                       unique_ptr,
                       )

# classes
from ROOT import (
                   TObjArray,
                   TBranch,
                   TBranchRef,
                   TKey,
                   TMemFile,
                   TTree,
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
from ctypes import (
                     c_double,
                     c_longlong,
)
Long64_t = c_longlong


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
                   gStyle,
                   gPad,
                   gRandom,
                   gBenchmark,
                   gROOT,
)




# #
# # Macro prototype.
# #
# # Summary of available functions. In .C version they are polymorphic.
# #
# def GetBasketSize      ( branches : TObjArray, ondisk : bool, inclusive : bool ) : pass
# def GetBasketSize      ( b : TBranch, ondisk : bool, inclusive : bool          ) : pass
# #
# def GetTotalSize       ( br : TBranch, ondisk : bool, inclusive : bool         ) : pass
# def GetTotalSize       ( branches : TObjArray, ondisk : bool                   ) : pass
# def GetTotalSize       ( t : TTree, ondisk : bool                              ) : pass
# #
# def sizeOnDisk         ( t : TTree                                             ) : pass
# def sizeOnDisk         ( branch : TBranch, inclusive : bool                    ) : pass
# #
# def printBranchSummary ( br : TBranch                                          ) : pass
# #
# def printTreeSummary   ( t : TTree                                             ) : pass
# #


@singledispatch
def GetBasketSize( ) : raise NotImplemented( "It requires proper arguments")

@singledispatch
def GetTotalSize( ) : raise NotImplemented( "It requires proper arguments")

@singledispatch
def sizeOnDisk( ) : raise NotImplemented( "It requires proper arguments")



# Long64_t
@GetBasketSize.register
def _(branches : TObjArray, ondisk : bool, inclusive : bool) :

   result = 0                 # Long64_t
   n = branches.GetEntries()  # size_t
   #
   #for (size_t i = 0; i < n; ++i) {
   for i in range(0, n, 1):
      result += GetBasketSize( branches.At(i), ondisk, inclusive)
      # Note :
      #        branches.At( i ) # TBranch *
      
   return result
   

# Long64_t
@GetBasketSize.register
def _(b : TBranch, ondisk : bool, inclusive : bool) :

   result = 0  # Long64_t

   if (b)  :

      if (ondisk and b.GetZipBytes() > 0)  :
         result = b.GetZipBytes()
         
      else  :
         result = b.GetTotBytes()
         
      if (inclusive)  :
         result += GetBasketSize(b.GetListOfBranches(), ondisk, True)
         
      return result
      
   return result
   

# Long64_t
@GetTotalSize.register
def _(br : TBranch, ondisk : bool, inclusive : bool) :

   f = TMemFile("buffer", "CREATE")

   if (br.GetTree().GetCurrentFile())  :
      f.SetCompressionSettings(
         br.GetTree().GetCurrentFile().GetCompressionSettings()
         )
      
   #
   f.WriteObject(br, "thisbranch")
   key = f.GetKey("thisbranch") # TKey *
   size = Long64_t()
   #
   if (ondisk)  :
      size = key.GetNbytes()
      
   else  :
      size = key.GetObjlen()
      

   #
   return GetBasketSize(br, ondisk, inclusive)
   

# Long64_t
@GetTotalSize.register
def _(branches : TObjArray, ondisk : bool) :

   result  = 0                      # Long64_t
   n       = branches.GetEntries()  # size_t

   #
   #for (size_t i = 0; i < n; ++i) {
   for i in range(0, n, 1):
      result += GetTotalSize( branches.At(i)  , ondisk, True)
      sys.stderr( "After " , branches.At(i).GetName() , " " , result )
      # Note :
      #        branches.At( i ) # TBranch *
      
   return result
   

# Long64_t
@GetTotalSize.register
def _(t : TTree, ondisk : bool) :

   #
   key = TKey () # nullptr
   #
   if (t.GetDirectory())  :
      key = t.GetDirectory().GetKey(t.GetName()) # TKey *
      
   #
   ondiskSize  = 0  # Long64_t
   totalSize   = 0  # Long64_t
   #
   if (key)  :
      ondiskSize = key.GetNbytes()
      totalSize  = key.GetObjlen()
      
   else  :
      f = TMemFile("buffer", "CREATE")
      if (t.GetCurrentFile())  :
         f.SetCompressionSettings(t.GetCurrentFile().GetCompressionSettings())
         
      f.WriteTObject(t)
      key        = f.GetKey(t.GetName())
      ondiskSize = key.GetNbytes()
      totalSize  = key.GetObjlen()
      
   if (t.GetBranchRef())  :
      if (ondisk)  :
         ondiskSize += GetBasketSize(t.GetBranchRef(), True, True)
         
      else  :
         totalSize += GetBasketSize(t.GetBranchRef(), False, True)
         
      
   if (ondisk)  :

      # on disk = True  # inclusive  = True
      return ondiskSize + GetBasketSize(t.GetListOfBranches(),  True,  True)
      
   else  :
      # on disk = False  # inclusive  = True
      return totalSize + GetBasketSize(t.GetListOfBranches(),  False,  True)
      
   

# Long64_t
@sizeOnDisk.register
def _(t : TTree) :

   # Return the size on disk on this TTree.
   
   return GetTotalSize(t, True)
   

# Long64_t
@sizeOnDisk.register
def _(branch : TBranch, inclusive : bool) :

   """
   Return the size's disk of this branch.
   If 'inclusive' is true, it includes also the size
   of all its sub-branches.
   """
   
   return GetTotalSize(branch, True, inclusive)
   

# void
def printBranchSummary(br : TBranch) :

   #
   print( "The branch \"" ,
         br.GetName() ,
         "\" takes " ,
         sizeOnDisk(br,
            True) ,
         " bytes on disk\n",
        )

   #
   n = br.GetListOfBranches().GetEntries()  # size_t
   #
   #for (size_t i = 0; i < n; ++i) {
   for i in range(0, n, 1):
      #
      subbr = br.GetListOfBranches().At(i) # TBranch *
      #
      print( "  Its sub-branch \"" ,
            subbr.GetName() ,
            "\" takes " ,
            sizeOnDisk(subbr,
            True) ,
            " bytes on disk\n",
           )
      
   

# void
def printTreeSummary(t : TTree) :

   #
   print( "The TTree \"" ,
         t.GetName() ,
         "\" takes " ,
         sizeOnDisk(t) ,
         " bytes on disk\n",
        )

   #
   n = t.GetListOfBranches().GetEntries()  # size_t
   #
   #for (size_t i = 0; i < n; ++i) {
   for i in range(0, n, 1):
      #
      br = t.GetListOfBranches().At(i) # TBranch *
      #
      print( "  Its branch \"" ,
            br.GetName() ,
            "\" takes " ,
            sizeOnDisk(br,
            True) ,
            " bytes on disk\n",
           )
      
      #
   


def printSizes():
   # Accessing a simple *.root file from the root/tutorials.
   global file
   file = TFile( "../legacy/mlp/mlpHiggs.root" )
   #
   # Get a Tree.
   global tree
   tree = file.Get( "bg_filtered" )
   # Or.
   # t = f.Get( "sig_filtered" )
   #
   # Choose one of the brancehs from the list produced by:
   # t.Print()
   # Let's say:
   global b_ptsumf
   b_ptsumf = tree.GetBranch( "ptsumf")
   b_acolin = tree.GetBranch( "acolin" )
   # b_acopl  =  tree.GetBranch( "acopl" )
   # b_minvis = tree.GetBranch( "minvis" )
   # b_msumf  =  tree.GetBranch( "msumf" )
   # b_nch    =    tree.GetBranch( "nch" )
   # b_qelep  =  tree.GetBranch( "qelep" )
   # b_ptsumf = tree.GetBranch( "ptsumf" )

   # Now, we can use to analyze how much space does it occupe 
   # on memory:
   # 
   print( "\nFor the root/tutorials/mlp/mplHiggs.root file : " )
   #
   print( "\nBranch Summary: \"bg_filtered / ptsumf\" " )
   printBranchSummary( b_ptsumf )
   #
   print( "\nBranch Summary: \"bg_filtered / b_acolin\" " )
   printBranchSummary( b_acolin )
   #
   print( "\nTree Summary: \"bg_filtered\" " )
   printTreeSummary( tree ) 
   
   #  Another example would be 
   try : 
      global file2, tree2
      file2             = TFile( "./tree4.root" )
      tree2             = file2.Get( "t4" )
      b_event_split     = tree2.GetBranch( "event_split" )
      b_event_not_split = tree2.GetBranch( "event_not_split" )
   except:
      import sys
      sys.exit()
      pass
   print( " \n >>> Another example :")
   print( "\nFor the root/tutorials/tree/tree4.root file : " )
   #
   print( "\nBranch Summary: \"\" " )
   printBranchSummary( b_event_split )
   #
   print( "\nBranch Summary: \"\" " )
   printBranchSummary( b_event_not_split )
   #
   print( "\nTree Summary: \"\" " )
   printTreeSummary( tree2 ) 








if __name__ == "__main__":
   printSizes()

