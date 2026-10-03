## \file
## \ingroup tutorial_spectrum
## \notebook
##
## Example illustrating a high resolution peak searching function 
## using the TSpectrum2 class.
##
## \macro_image
## \macro_output
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
                   TMarker,
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

   rows, cols = len( matrix ), len( matrix[0] )

   data = ( c_double * ( rows * cols ) )()
   row_pointers = ( POINTER( c_double ) * rows )()
   
   # fill data
   for i in range(rows) :
      for j in range(cols) :
         data[ i*cols + j ] = matrix[ i ][ j ] 
  
   # fill addresses
   for i in range( rows ) :
      # Ok
      #row_pointers[i] =\
      #   cast( byref( data, i * cols * sizeof(c_double) ), POINTER(c_double) )

      # Ok
      row_pointers[i] =\
         ( c_double * cols ).from_buffer( data, i*( cols * sizeof( c_double) ) ) 


   return row_pointers



# void
def Src5() :

   nbinsx = 64
   nbinsy = 64


   ##std.vector<Double_t *> source(nbinsx), dest(nbinsx)
   # 
   ##for (Int_t i = 0; i < nbinsx; i++) {
   #for i in range(0, nbinsx, 1):
   #   source[i] = new Double_t[nbinsy];
   #   dest[i]   = new Double_t[nbinsy];
   #
   source = [ [ float() for _ in range( nbinsy ) ] for _ in range( nbinsx ) ] 
   dest   = [ [ float() for _ in range( nbinsy ) ] for _ in range( nbinsx ) ]
      
   Dir  = gROOT.GetTutorialDir()                        # TString
   file = Dir + TString( "/spectrum/TSpectrum2.root" )  # TString
   global f
   f = TFile.Open( file.Data() ) # TFile *

   gStyle.SetOptStat( 0 )

   global search
   search = f.Get("search3;1") # (TH2F *)


   # # #
   global s
   s = TSpectrum2()
   #for (Int_t i = 0; i < nbinsx; i++) {
   for i in range(0, nbinsx, 1):
      #      for (Int_t j = 0; j < nbinsy; j++) {
      for j in range(0, nbinsy, 1):
         source[i][j] = search.GetBinContent( i + 1, j + 1 )
         
      
   # double** type
   global source_ptr_ptr, dest_ptr_ptr
   source_ptr_ptr = to_c_double_ptr_ptr_FLAT( source )  
   dest_ptr_ptr   = to_c_double_ptr_ptr_FLAT( dest   )  

   global nfound
   nfound = s.SearchHighRes(
                             source_ptr_ptr ,
                             dest_ptr_ptr   ,
                             nbinsx         ,
                             nbinsy         ,
                             2              ,
                             5              ,
                             False          ,
                             10             ,
                             False          ,
                             1              ,
                             ) # int

   print( "Found %d candidate peaks\n" % nfound )

   
   global PositionX, PositionY
   PositionX = s.GetPositionX() # Double_t *
   PositionY = s.GetPositionY() # Double_t *

   search.Draw( "COL" )

   global m
   m = TMarker()
   m.SetMarkerStyle ( 23   )
   m.SetMarkerColor ( kRed )

   #for (Int_t i = 0; i < nfound; i++) {
   for i in range(0, nfound, 1):
      print( 
             "posx= %d, posy= %d, value=%d\n" % (
                int( PositionX[i] + 0.5 )                       ,
                int( PositionY[i] + 0.5 )                       ,
                int( source[                            \
                            int( PositionX[i] + 0.5 )   \
                            ]                           \
                           [                            \
                            int( PositionY[i] + 0.5 )   \
                            ]
                    )                                           ,
                )
            )
      m.DrawMarker( PositionX[i], PositionY[i] )
      
   
   del source
   del source_ptr_ptr
   del dest_ptr_ptr
   del dest
      
   


if __name__ == "__main__":
   Src5()
