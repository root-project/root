## \file
## \ingroup tutorial_tree
## \notebook
##
##
## About the integration of TTree with THnSparse and TParallelCoord 
## classes.
##
## This script converts a THnSparse object to a TTree object using efficient
## iteration through the THnSparse methods,
## and draws a THnSparse object using the TParallelCoord class.
##
## The resulting plot will contain one line for each filled bin,
## with the bin's coordinates on each axis, and the bin's content on
## the rightmost axis.
##
## Run as:
## ~~~
##    IPython [1]: drawsparse_script = !echo $ROOTSYS/tutorials/tree/drawsparse.py
##    IPython [2]: %run drawsparse_script[0]
## ~~~
##
## \macro_image
## \macro_code
##
## \author Axel Naumann
## \translator P. P.


import numpy as np
import ctypes
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
                   THnSparseD,
                   TIter,
                   TAxis,
                   TCanvas,
                   TFile,
                   TH3,
                   THnSparse,
                   TLeaf,
                   TParallelCoord,
                   TParallelCoordVar,
                   TROOT,
                   TRandom,
                   TTree,
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
                   kOrange,
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




# TTree
def toTree(h : THnSparse) :

   """
   Creates a TTree and fills it with the coordinates of all
   filled bins. 
   The resulting tree will have one branch for each dimension,
   and another branch for the bin content.

   """
   
   
   #
   dim    =  h.GetNdimensions()  #  Int_t
   #
   name   =  h.GetName()         #  TString
   name  +=  "_tree"
   #
   title  =  h.GetTitle()        #  TString
   title +=  " tree"
   #
   tree = TTree(name, title)  # TTree


   #
   # Inherited from the .C version:
   # x = np.array( x, dtype="d")       # Double_t *
   # x = np.zeros( dim + 1, dtype="d")
   #
   # Two different names for the two branches is recommended.
   x = np.zeros( dim , dtype="d")
   x_bin = np.zeros( 1, dtype="d")

   
   #
   # Construct the branch name for the types (n-dim):
   # like : 
   #         ":x1/D:x2/D:x3/D..." 
   #
   branchname = "" # TString()
   #
   #for (Int_t d = 0; d < dim; ++d) {
   for d in range(0, dim, 1):
      #
      #if (branchname.Length())  :
      if len( branchname ) :
         branchname += ":"
         
      #
      axis = h.GetAxis(d) # TAxis *
      #
      branchname  += axis.GetName() # TString
      branchname  += "/D"
   #
   # print( branchname )
   # Output:
   # axis0/D:axis1/D:axis2/D:axis3/D:axis4/D:axis5/D:axis6/D:axis7/D

     
   #
   # Not to use:
   #tree.Branch("coord"      , x[:dim]      , branchname    )
   #tree.Branch("bincontent" , x[dim] , "bincontent/D"      )  
   #
   # Instead:
   tree.Branch("coord"      , x      , branchname     )
   tree.Branch("bincontent" , x_bin , "bincontent/D" )  
   # Note :
   #        In the .C version bincontent and all the axes
   #        are stucked in one variable "x".
   #        In pyroot, we cannot slice a numpy array
   #        with the `[ : ]` notation to create two different 
   #        branches, because unexpected numerical errors arise.
   #        The plot limits of "bincontent", the last axis,
   #        go around 1000000E23 values. A clearly
   #        memory leak problem. C-pointer arrays are not 
   #        numpy arrays, sometimes they behave likewise, 
   #        sometimes they don't.
   #        Defining different array variables for different
   #        branches is recommended.
   #        
   #         
   
   
   #
   # Indexes.
   bins = np.zeros( dim, "i" ) # Int_t[dim] 
   #
   #for (Long64_t i = 0; i < h->GetNbins(); ++i) {
   for i in range( 0, h.GetNbins(), 1 ) : 
      #
      # Fill the last axis and fill the bins indexes.
      x_bin[0] = h.GetBinContent(i, bins) # Double_t
      #
      # Fill the first seven axis.
      #for (Int_t d = 0; d < dim; ++d) {
      for d in range(0, dim, 1):
         #  
         _bin_ = bins[d].item()
         x[d] = h.GetAxis(d).GetBinCenter( _bin_ )  # Double_t
         #
         # REVIEW:
         #         TAxis::GetBinCenter( Int_t bin ).
         #         Shouldn't it be:
         #            TAxis::GetBinCenter( Int_t & bin )
         #            TAxis::SetBinCenter( Int_t bin )
         #         Could there be a semantics confusion.
         
      
      #
      tree.Fill()
      
   
   #
   del bins              # []
   del x                 # delete [] x

   #  
   return tree
   

# void
def drawsparse_draw(h : THnSparse) :

   """
   Draw a THnSparse using TParallelCoord, creating a temporary TTree.

   """
   # - - - 1 - - -

   #
   tree = toTree(h) # TTree *
   
   #
   whatToDraw = "" # TString()

   iLeaf = TIter( tree.GetListOfLeaves() )
   leaf = nullptr # TLeaf * 
   #
   # while ((leaf = (const TLeaf *)iLeaf()))  :
   for leaf in iLeaf :
      #
      #if (whatToDraw.Length())  :
      if len( whatToDraw ) : 
         whatToDraw += ":"
      #   
      whatToDraw += leaf.GetName()
   #
   # print( "whatToDraw: \n", whatToDraw )
   # Output: 
   #         whatToDraw:  
   #         axis0:axis1:axis2:axis3:axis4:axis5:axis6:axis7:bincontent


   #
   tree.Draw( whatToDraw, "", "para")


   # - - - 2 - - -

   #
   # ParaCoord exists temporarely due to 
   # the way we use .Draw along with the "para" argument.
   parallelCoord = \
      gPad.GetListOfPrimitives().FindObject(
         "ParaCoord",
         )  # (TParallelCoord *)
  


   # 
   # - - - 3 - - -
   # 
   global iVar, var
   iVar = TIter( parallelCoord.GetVarList() )
   var = nullptr # TParallelCoordVar *
   # 
   # Too large :
   # for (Int_t d = 0; (var = (TParallelCoordVar *)iVar()) && d < h->GetNdimensions(); ++d)  :
   # for d in range( 0, h.GetNDimensions(), 1 ) 
       # var = iVar.Next() # * TParallelCoordVar
       # ...
   # var = iVar.Next() 
   # ...
   #
   # Instead :
   d = 0
   for var in iVar: 
      if d == h.GetNdimensions() : 
         break
      #
      axis = h.GetAxis(d) # TAxis *
      #
      var.SetCurrentLimits    ( axis.GetXmin  ( ), axis.GetXmax ( ) )
      var.SetTitle            ( axis.GetTitle ( )                   )
      var.SetHistogramBinning ( axis.GetNbins ( )                   )
      #
      d += 1
   #   
   # The last one contains the histogram of bins.
   var.SetTitle("bin content")
   #
   # Note: The last axis' labels are set 
   #       automatically.

   #
   # Note:
   #       In [  ]: for var in iVar:
   #           ...:     print(var)
   #           ...: 
   #           ...: 
   #       Out[  ]:  Name: axis1 Title: axis1
   #                 Name: axis2 Title: axis2
   #                 Name: axis3 Title: axis3
   #                 Name: axis4 Title: axis4
   #                 Name: axis5 Title: axis5
   #                 Name: axis6 Title: axis6
   #                 Name: axis7 Title: axis7
   #                 Name: bincontent Title: bincontent
   #
   # The last one is different. Ergo, the previous loop was unusual.


   

# void
def drawsparse() :

   """
   Creates a THnSparse and draws it.

   """

   #
   ndims = 8 # Int_t
   #
   #         |     |       |       |    |    |     |   |  
   bins = [ 10  , 10   , 5      , 30  , 10 , 4  , 18 , 12 ]  # Int_t    [ndims]
   xmin = [ -5. , -10. , -1000. , -3. , 0. , 0. , 0. , 0. ]  # Double_t [ndims]
   xmax = [ 10. , 70.  , 3000.  , 3.  , 5. , 2. , 2. , 5. ]  # Double_t [ndims]
   #         |     |       |       |    |    |     |   |  
   #
   bins = np.array( bins, dtype="i" )
   xmin = np.array( xmin, dtype="d" )
   xmax = np.array( xmax, dtype="d" )
   

   #
   global hs
   hs = THnSparseD(
                    "hs",
                    "Sparse Histogram",
                    ndims,
                    bins,
                    xmin,
                    xmax,

                    )  # THnSparse

   
   #
   # Fill it.
   #
   x = np.zeros( ndims, dtype="d" ) # Double_t *
   #
   #for (Long_t i = 0; i < 100000; ++i) {
   for i in range(0, 100000, 1):
   #for i in range(0, 100, 1): # Test.
      #
      #      for (Int_t d = 0; d < ndims; ++d) {
      for d in range(0, ndims, 1):
         #
         match (d) :
           case 0:
              x[d] = gRandom.Gaus() * 2 + 3.

           case 1 | 2 | 3: 
              x[d] = (x[d - 1] * x[d - 1] - 1.5) / 1.5 + (0.5 * gRandom.Rndm())

           case _ :
              x[d] = sin(gRandom.Gaus() * i / 1000.) + 1.
            
      #   
      hs.Fill(x)
      

   #
   # Notice, we don't write the THnSparseT object "hs" in 
   # the root file. As mentioned early, this method
   # only needs a temporary tree. The trick lies on
   # the TTree.Draw method, which we'll use in the
   # "drawsparse_draw" function later.
   
   #
   global f
   f = TFile("drawsparse.root", "RECREATE")  # TFile
   
   #
   global canv
   canv = TCanvas("hDrawSparse", "Drawing a sparse hist")  # TCanvas
   canv.Divide(2)

   
   #
   # Draw it.
   #
   canv.cd(1)
   #
   drawsparse_draw( hs )

   
   #
   # Project it on 3D.
   #
   h3proj = hs.Projection(2, 3, 6) # TH3D *
   #
   h3proj.SetLineColor(kOrange)
   h3proj.SetDirectory(nullptr)
   #
   canv.cd(2)
   h3proj.Draw("lego1")

   #
   canv.Update()
   canv.Draw()
   
   #
   # Save everything into the file "drawsparse.root".
   #
   canv.Write()
   hs.Write()
   h3proj.Write()
   
   #
   f.Close()
   gROOT.Remove( f )
   del f

   # 
   #hDrawSparse = gROOT.FindObject( "hDrawSparse" )
   #gROOT.Remove( hDrawSparse )

   


if __name__ == "__main__":
   drawsparse()
