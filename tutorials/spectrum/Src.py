## \file
## \ingroup tutorial_spectrum
## \notebook
##
## Example to illustrate a high resolution peak searching function 
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
                   TMarker,
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
                     sizeof,
                     POINTER,
                     )

#utils
def to_c( ls ):
   return (c_double * len(ls) )( * ls )

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
         ( c_double * cols ).from_buffer( data, i*( cols * sizeof( c_double ) ) ) 


   return row_pointers



# void
def Src() :

   nbinsx = 64
   nbinsy = 64


   # TODO: pythonize .data() LowLevelView of std.vector properly for ["double*"] type.
   #
   # Note: std.vector[ "double * "] doesn't have the method .data() properly pythonized.
   #       A simple list of lists will do the trick in python. 
   #
   # C++:
   #std.vector<Double_t *> source(nbinsx), dest(nbinsx)
   # Python:
   #source = std.vector[ "Double_t *" ]( nbinsx )
   #dest   = std.vector[ "Double_t *" ]( nbinsx )
   #
   ##for (Int_t i = 0; i < nbinsx; i++) {
   #for i in range(0, nbinsx, 1):
   #   source[i] =  Double_t[nbinsy] # new
   #   dest[i]   =  Double_t[nbinsy] # new
   #
   source = [ [ float() for _ in range( nbinsy ) ] for _ in range( nbinsx ) ] 
   dest   = [ [ float() for _ in range( nbinsy ) ] for _ in range( nbinsx ) ] 
      
   Dir  = gROOT.GetTutorialDir()                        # TString
   file = Dir + TString( "/spectrum/TSpectrum2.root" )  # TString
   global f
   f = TFile.Open( file.Data() ) # TFile *

   gStyle.SetOptStat( 0 )
 
   global search
   search = f.Get( "search4" ) # (TH2F *)


   # # #
   global s
   s = TSpectrum2()


   #for (Int_t i = 0; i < nbinsx; i++) {
   for i in range(0, nbinsx, 1):
      #      for (Int_t j = 0; j < nbinsy; j++) {
      for j in range(0, nbinsy, 1):
         source[i][j] = search.GetBinContent( i + 1, j + 1 )
         
   global source_ptr_ptr, dest_ptr_ptr
   source_ptr_ptr = to_c_double_ptr_ptr_FLAT( source )
   dest_ptr_ptr   = to_c_double_ptr_ptr_FLAT( dest   )

   # Int_t
   global nfound
   nfound = s.SearchHighRes(
                            source_ptr_ptr,
                            dest_ptr_ptr,
                            nbinsx,
                            nbinsy,
                            2,
                            5,
                            True,
                            3,
                            False,
                            3,
                            )

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
             "posx= %d, posy= %d, value=%d\n"             ,
             int( PositionX[i] + 0.5 )                    ,
             int( PositionY[i] + 0.5 )                    ,
             int( source[                            \
                         int( PositionX[i] + 0.5 )   \
                         ]                           \
                        [                            \
                         int( PositionY[i] + 0.5 )   \
                         ] 
                 )                                        ,
            )
      m.DrawMarker( PositionX[i], PositionY[i] )
      
   
   del source, dest
   del source_ptr_ptr, dest_ptr_ptr
      
   


if __name__ == "__main__":
   Src()
