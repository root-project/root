## \file
## \ingroup tutorial_tree
## \notebook -nodraw
##
##
## Script illustrating the use of the TParalleCoord class.
##
##
## \macro_image
## \macro_code
##
## \author  Bastien Dallapiazza
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
                   TCanvas,
                   TFile,
                   TNtuple,
                   TParallelCoord,
                   TParallelCoordRange,
                   TParallelCoordVar,
                   TRandom,
                   TRandom3,
                   TStyle,
                   TMath,
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
                   kBlack,
                   kOrange,
                   kViolet,
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
#r1, r2, r3, r4, r5, r6, r7, r8, r9 = ( np.zeros(1, "d") for _ in range( 9 ) )  # Double_t
r1, r2, r3, r4, r5, r6, r7, r8, r9 = ( c_double() for _ in range( 9 ) )  # Double_t
dr = 3.5        # Double_t 
r = TRandom()   # TRandom * # error on high numbers ~10000.
r = TRandom3()   # TRandom *


#
def TRandom_Sphere( random_gen , radius = 1.):
   
   #random_gen = TRandom() # TRandom3 # TRandom2  
   
   # Generate random spherical coordinates.
   theta = random_gen.Uniform(0, 2 * TMath.Pi() )  # Azimuthal angle
   phi   = random_gen.Uniform(0, TMath.Pi()     )  # Polar angle
   
   # Convert spherical coordinates to Cartesian coordinates.
   x = radius * np.sin(phi) * np.cos(theta)
   y = radius * np.sin(phi) * np.sin(theta)
   z = radius * np.cos(phi)
   
   return x, y, z 
   
# void
def generate_random(i : Int_t) :
   #
   # r1[0] = (2 * dr * r.Rndm(i)) - dr
   # r2[0] = (2 * dr * r.Rndm(i)) - dr
   # r7[0] = (2 * dr * r.Rndm(i)) - dr
   # r9[0] = (2 * dr * r.Rndm(i)) - dr
   # r4[0] = (2 * dr * r.Rndm(i)) - dr
   # r3[0] = (2 * dr * r.Rndm(i)) - dr
   # r5[0] = (2 * dr * r.Rndm(i)) - dr
   # r6[0] = (2 * dr * r.Rndm(i)) - dr
   # r8[0] = (2 * dr * r.Rndm(i)) - dr

   r1.value = (2 * dr * r.Rndm(i)) - dr
   r2.value = (2 * dr * r.Rndm(i)) - dr
   r7.value = (2 * dr * r.Rndm(i)) - dr
   r9.value = (2 * dr * r.Rndm(i)) - dr
   r4.value = (2 * dr * r.Rndm(i)) - dr
   r3.value = (2 * dr * r.Rndm(i)) - dr
   r5.value = (2 * dr * r.Rndm(i)) - dr
   r6.value = (2 * dr * r.Rndm(i)) - dr
   r8.value = (2 * dr * r.Rndm(i)) - dr
   

# void
def parallelcoord() :
   
   # #
   # global nt
   # nt = TNtuple() # nullptr # TNtuple *
   
   #
   #s1x, s1y, s1z = ( np.zeros(1, "d") for _ in range(3) )  # Double_t
   #s2x, s2y, s2z = ( np.zeros(1, "d") for _ in range(3) )  # Double_t
   #s3x, s3y, s3z = ( np.zeros(1, "d") for _ in range(3) )  # Double_t
   s1x, s1y, s1z = ( c_double() for _ in range(3) )  # Double_t
   s2x, s2y, s2z = ( c_double() for _ in range(3) )  # Double_t
   s3x, s3y, s3z = ( c_double() for _ in range(3) )  # Double_t
   #
   global r
   # r =  TRandom()  # new # #
   
   
   #
   global new_canvas
   new_canvas = TCanvas("c1", "c1", 0, 0, 800, 700)  # new
   
   #
   global nt
   nt =  TNtuple("nt", "Demo ntuple", "x:y:z:u:v:w")  # new
   

   # Note :
   #        TRandom.Sphere generates error on loop over ~1000 .
   #        TRandom3.Sphere generates error on loop over ~1000 .
   #        There is some interference with the automatic
   #        python garbage collector.
   #        
   #
   #for (Int_t i = 0; i < 20000; i++) {
   for i in range(0, 20000, 1): # error # no-error if we use TRandom_Sphere.
   #for i in range(0, 1000, 1):  # ok
      #
      # print( "i:", i ) # Look when crashes.

      #
      # Works fine for lower values in the loop.
      # r.Sphere(s1x, s1y, s1z, 0.1)
      # r.Sphere(s2x, s2y, s2z, 0.2)
      # r.Sphere(s3x, s3y, s3z, 0.05)
       
      # No error.
      s1x.value, s1y.value, s1z.value = TRandom_Sphere(r, 0.1)
      s2x.value, s2y.value, s2z.value = TRandom_Sphere(r, 0.2)
      s3x.value, s3y.value, s3z.value = TRandom_Sphere(r, 0.05)
      # "r" is the randon generator.
      # TODO: 
      #       Syntax is large. Shorten. 

      
      generate_random(i)
      nt.Fill(r1.value,
              r2.value,
              r3.value,
              r4.value,
              r5.value,
              r6.value,
              )
      
      generate_random(i)
      nt.Fill( s1x.value,
               s1y.value,
               s1z.value,
               s2x.value,
               s2y.value,
               s2z.value,
               )
      
      generate_random(i)
      nt.Fill( r1.value,
               r2.value,
               r3.value,
               r4.value,
               r5.value,
               r6.value,
               )
      
      generate_random(i)
      nt.Fill( s2x.value - 1,
               s2y.value - 1,
               s2z.value,
               s1x.value + .5,
               s1y.value + .5,
               s1z.value + .5,
               )
      
      generate_random(i)
      nt.Fill( r1.value,
               r2.value,
               r3.value,
               r4.value,
               r5.value,
               r6.value,
               )
      
      generate_random(i)
      nt.Fill( s1x.value + 1,
               s1y.value + 1,
               s1z.value + 1,
               s3x.value - 2,
               s3y.value - 2,
               s3z.value - 2,
               )
      
      generate_random(i)
      nt.Fill( r1.value,
               r2.value,
               r3.value,
               r4.value,
               r5.value,
               r6.value,
               )
      
   #
   nt.Draw("x:y:z:u:v:w", "", "para", 5000)

   #
   para = gPad.GetListOfPrimitives().FindObject("ParaCoord")  # (TParallelCoord *)
   para.SetDotsSpacing(5)

   #
   firstaxis = para.GetVarList().FindObject("x")  # (TParallelCoordVar *)

   # - a - 
   # para.AddSelection("black")
   # para.GetCurrentSelection().SetLineColor(kBlack)
   new_ParallelCoordRange1 = TParallelCoordRange(firstaxis, 0.846018, 1.158469)
   firstaxis.AddRange( new_ParallelCoordRange1 ) 

   #
   # - b - 
   para.AddSelection("violet")
   para.GetCurrentSelection().SetLineColor(kViolet)
   #
   new_ParallelCoordRange2 = TParallelCoordRange(firstaxis, -0.169447, 0.169042)
   firstaxis.AddRange( new_ParallelCoordRange2 )

   #
   # - c - 
   para.AddSelection("Orange")
   para.GetCurrentSelection().SetLineColor(kOrange + 9)
   #
   new_ParallelCoordRange3 = TParallelCoordRange(firstaxis, -1.263024, -0.755292)
   firstaxis.AddRange( new_ParallelCoordRange3 )
   


if __name__ == "__main__":
   parallelcoord()
