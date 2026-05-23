## \file
## \ingroup tutorial_spectrum
## \notebook
## Example to illustrate the Markov smoothing (class TSpectrum2).
##
## \macro_image
## \macro_code
##
## \authors Miroslav Morhac, Olivier Couet
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
                   TSpectrum2,
                   TFile,
                   TString,
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

# ctypes
from ctypes import (
                     c_double,
                     POINTER,
                     sizeof,
                     byref,
                     )

# utils
def to_c( ls ):
   return ( c_double * len( ls ) )( * ls )

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


# For C++ type: double **
def to_c_double_ptr_ptr_FLAT( matrix ) :

   rows, cols = len( matrix ), len( matrix[0])

   data = ( c_double * ( rows * cols ) )()
   row_pointers = ( POINTER( c_double ) * rows )()
   
   # fill data
   for i in range(rows) :
      for j in range(cols) :
         data[ i*cols + j ] = matrix[ i ][ j ] 
  
   # fill addresses
   for i in range( rows ) :
      # Ok
      # row_pointers[i] =\
      #   cast( byref( data, i * cols * sizeof(c_double) ), POINTER(c_double) )

      # Ok
      row_pointers[i] =\
         ( c_double * cols ).from_buffer( data, i*( cols * sizeof( c_double ) ) ) 


   return row_pointers


# void
def Smooth() :

   # Error:
   #nbinsx = 256
   #nbinsy = 256
   # Error:
   #nbinsx = 200
   #nbinsy = 200
   # Ok
   nbinsx = 150
   nbinsy = 150
   # Ok
   #nbinsx = 100
   #nbinsy = 100
   #
   xmin = 0.
   xmax = float( nbinsx )
   #
   ymin = 0.
   ymax = float( nbinsy )
   
   # FROM C++:
   #
   # Double_t ** source =  Double_t * [nbinsx]; # new
   # #for (i = 0; i < nbinsx; i++) {
   # for i in range(0, nbinsx, 1):
   #    source[i] =  Double_t[nbinsy]; # new
   #
   # TO PYTHON:
   # [ x ] [ y ]
   global source
   source = [ [ float() for _ in range( nbinsy ) ] for _ in range( nbinsx ) ] 

      
   Dir  = gROOT.GetTutorialDir()                        # TString
   file = Dir + TString( "/spectrum/TSpectrum2.root" )  # TString
   global f
   f = TFile( file.Data() ) # TFile

   global smooth
   smooth = f.Get( "smooth1" ) # (TH2F *)

   gStyle.SetOptStat( 0 )


   # # #
   global s
   s = TSpectrum2()


   #for (i = 0; i < nbinsx; i++) {
   for i in range(0, nbinsx, 1):
      #      for (j = 0; j < nbinsy; j++) {
      for j in range(0, nbinsy, 1):
         source[i][j] = smooth.GetBinContent( i + 1, j + 1 )
         

   # to double** type
   global source_ptr_ptr
   source_ptr_ptr   = to_c_double_ptr_ptr_FLAT( source )
      
   # # #
   s.SmoothMarkov(
                   source_ptr_ptr,
                   nbinsx,
                   nbinsx,

                   # 1 or 3 or 5 or 7 # Whichever works fine.
                   3,
                   ) 



   #for (i = 0; i < nbinsx; i++) {
   for i in range(0, nbinsx, 1):
      #      for (j = 0; j < nbinsy; j++) {
      for j in range(0, nbinsy, 1):
         smooth.SetBinContent(
                               i + 1                ,
                               j + 1                ,
                               source_ptr_ptr[i][j] ,
                               )
         
      
   smooth.Draw( "SURF2" )
   


if __name__ == "__main__":
   Smooth()
