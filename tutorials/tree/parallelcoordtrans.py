## \file
## \ingroup tutorial_tree
## \notebook
##
##
## Use of transparency with parallel coordinates.
##
## It displays the same data set twice. The first time without transparency and
## the second time with transparency. On the second plot, several clusters
## appear.
## 
## It will help you at identify patterns, correlations, or clusters in your data.
## This technique is highly recommended for high-dimensional data; with it 
## you can plot n-dim data.
## In the plot, each point( n.dim ) lays on a parallel set of vertical lines, where
## each line represents a variable of the n-dimensional data.
##
## ### Images without and with transparency
##
## \macro_image
##
## ### Transparency works in PDF files
##
## \macro_image (parallelcoordtrans.pdf)
##
## \macro_code
##
## \author Olivier Couet
## \translator P. P.


import numpy as np
from numpy import float64 as np_float64
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
                   TCanvas,
                   TFile,
                   TNtuple,
                   TParallelCoord,
                   TParallelCoordRange,
                   TParallelCoordVar,
                   TRandom,
                   TStyle,
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




# variables
#
# Random numbers.
r1, r2, r3, r4, r5, r6, r7, r8, r9  = ( np.float64() for _ in range(9) ) # Double_t 
#
# Constant step for random variables.
dr = 3.5                            # Double_t 
#
# One random generator for this numbers.
r = TRandom() # *


# void
def generate_random(i : Int_t) :
   global r
   global r1, r2, r3, r4, r5, r6, r7, r8, r9

   #
   r.Rannor(r1, r4)
   r.Rannor(r7, r9)

   #
   r2 = (2 * dr * r.Rndm(i)) - dr
   r3 = (2 * dr * r.Rndm(i)) - dr
   r5 = (2 * dr * r.Rndm(i)) - dr
   r6 = (2 * dr * r.Rndm(i)) - dr
   r8 = (2 * dr * r.Rndm(i)) - dr
   

# void
def parallelcoordtrans() :

   #
   # Cartesian Coordinates.
   x, y, z, u, v, w, a, b, c = ( np.float64() for _ in range( 9 ) ) 
   #
   # Spherical Coordinates.
   s1x, s1y, s1z = ( np.float64() for _ in range( 3 ) ) 
   s2x, s2y, s2z = ( np.float64() for _ in range( 3 ) ) 
   s3x, s3y, s3z = ( np.float64() for _ in range( 3 ) ) 
   #
   # Their random generator... was defined on top.
   # r   = TRandom()  # new
   
   #
   global c1
   c1 =  TCanvas("c1", "c1", 0, 0, 900, 1000); # auto # new
   #
   c1.Divide(1, 2)

   # 
   global nt
   nt =  TNtuple("nt", "Demo ntuple", "x:y:z:u:v:w:a:b:c"); # auto # new
   
   # Inner counter for nTuple generation.
   n = 0  # int
   #
   #for (Int_t i = 0; i < 1500; i++) {
   #
   #for i in range(0, 1500, 1): # Poor visualization.
   #for i in range(0, 150, 1):  # Still no good visualization.
   for i in range(0, 50, 1):    # Good visualization.
      #
      r.Sphere(s1x, s1y, s1z, 0.1)
      r.Sphere(s2x, s2y, s2z, 0.2)
      r.Sphere(s3x, s3y, s3z, 0.05)
      
      #
      generate_random(i)
      nt.Fill(r1, r2, r3, r4, r5, r6, r7, r8, r9)
      n += 1
      
      #
      generate_random(i)
      nt.Fill(s1x, s1y, s1z, s2x, s2y, s2z, r7, r8, r9)
      n += 1
      
      #
      generate_random(i)
      nt.Fill(r1, r2, r3, r4, r5, r6, r7, s3y, r9)
      n += 1
      
      #
      generate_random(i)
      nt.Fill(s2x - 1, s2y - 1, s2z, s1x + .5, s1y + .5, s1z + .5, r7, r8, r9)
      n += 1
      
      #
      generate_random(i)
      nt.Fill(r1, r2, r3, r4, r5, r6, r7, r8, r9)
      n += 1
      
      #
      generate_random(i)
      nt.Fill(s1x + 1, s1y + 1, s1z + 1, s3x - 2, s3y - 2, s3z - 2, r7, r8, r9)
      n += 1
      
      #
      generate_random(i)
      nt.Fill(r1, r2, r3, r4, r5, r6, s3x, r8, s3z)
      n += 1
      
   
   #
   # global pcv
   # pcv = TParallelCoordVar() 
   #

   
   # 
   c1.cd(1)
   
   #
   # Parallel coordinates plot without transparency.
   #
   nt.Draw("x:y:z:u:v:w:a:b:c", "", "para")
   #
   global para1
   para1 = gPad.GetListOfPrimitives().FindObject("ParaCoord")  # (TParallelCoord *)
   #para1.SetLineColor(25)
   para1.SetLineColor(42)
   #
   global pcv
   pcv = [ ]
   pcv.append( para1.GetVarList().FindObject("x") ) # (TParallelCoordVar *)
   pcv[ -1 ].SetHistogramHeight(0.)
   #
   pcv.append( para1.GetVarList().FindObject("y") ) # (TParallelCoordVar *)
   pcv[ -1 ].SetHistogramHeight(0.)
   #
   pcv.append( para1.GetVarList().FindObject("z") ) # (TParallelCoordVar *)
   pcv[ -1 ].SetHistogramHeight(0.)
   #
   #
   pcv.append( para1.GetVarList().FindObject("a") ) # (TParallelCoordVar *)
   pcv[ -1 ].SetHistogramHeight(0.)
   #
   pcv.append( para1.GetVarList().FindObject("b") ) # (TParallelCoordVar *)
   pcv[ -1 ].SetHistogramHeight(0.)
   #
   pcv.append( para1.GetVarList().FindObject("c") ) # (TParallelCoordVar *)
   pcv[ -1 ].SetHistogramHeight(0.)
   #
   #
   pcv.append( para1.GetVarList().FindObject("u") ) # (TParallelCoordVar *)
   pcv[ -1 ].SetHistogramHeight(0.)
   #
   pcv.append( para1.GetVarList().FindObject("v") ) # (TParallelCoordVar *)
   pcv[ -1 ].SetHistogramHeight(0.)
   #
   pcv.append( para1.GetVarList().FindObject("w") ) # (TParallelCoordVar *)
   pcv[ -1 ].SetHistogramHeight(0.)
   #
   c1.Update()
   

   #
   # Parallel coordinates plot with transparency.
   #
   global col26 # Important for transparency!
   col26 = gROOT.GetColor(26) # TColor *
   col26.SetAlpha(0.01)
   #
   c1.cd(2)
   nt.Draw("x:y:z:u:v:w:a:b:c", "", "para")
   #
   global para2
   para2 = gPad.GetListOfPrimitives().FindObject("ParaCoord")  # (TParallelCoord *)
   para2.SetLineColor(26)
   #
   global pcv2
   pcv2 = [ ]
   pcv2.append( para2.GetVarList().FindObject("x") ) # (TParallelCoordVar *)
   pcv2[-1].SetHistogramHeight(0.)
   #
   pcv2.append( para2.GetVarList().FindObject("y") ) # (TParallelCoordVar *)
   pcv2[-1].SetHistogramHeight(0.)
   #
   pcv2.append( para2.GetVarList().FindObject("z") ) # (TParallelCoordVar *)
   pcv2[-1].SetHistogramHeight(0.)
   #
   #
   pcv2.append( para2.GetVarList().FindObject("a") ) # (TParallelCoordVar *)
   pcv2[-1].SetHistogramHeight(0.)
   #
   pcv2.append( para2.GetVarList().FindObject("b") ) # (TParallelCoordVar *)
   pcv2[-1].SetHistogramHeight(0.)
   #
   pcv2.append( para2.GetVarList().FindObject("c") ) # (TParallelCoordVar *)
   pcv2[-1].SetHistogramHeight(0.)
   #
   #
   pcv2.append( para2.GetVarList().FindObject("u") ) # (TParallelCoordVar *)
   pcv2[-1].SetHistogramHeight(0.)
   #
   pcv2.append( para2.GetVarList().FindObject("v") ) # (TParallelCoordVar *)
   pcv2[-1].SetHistogramHeight(0.)
   #
   pcv2.append( para2.GetVarList().FindObject("w") ) # (TParallelCoordVar *)
   pcv2[-1].SetHistogramHeight(0.)
   #
   c1.Update()
   
   #
   # Produce transparent lines in interactive and batch mode
   #
   c1.Print("parallelcoordtrans.pdf")
   c1.Print("parallelcoordtrans.svg")
   
   #
   # Produce transparent lines in batch mode only
   #
   c1.Print("parallelcoordtrans.jpg")
   c1.Print("parallelcoordtrans.png")
   


if __name__ == "__main__":
   parallelcoordtrans()
