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
                   TMarker,
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
                   Long_t,
                   Bool_t,
                   Float_t,
                   Int_t,
                   nullptr,
                   )

from ctypes import (
                     c_double,
                     POINTER,
                     sizeof,
                     byref,
                     cast,
                     )

#utils
def to_c( ls ):
   return ( c_double * len( ls ) )( * ls )

# constants
from ROOT import (
                   kOrange,
                   kMagenta,
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




# For c++ type: double**
def to_c_double_ptr_ptr(rows, cols) : 
   
   LP_c_double    = POINTER( c_double )
   LP_LP_c_double = POINTER( POINTER( c_double ) ) 
   
   c_double_Array_Array = ( c_double * cols ) * rows

   array_2d = c_double_Array_Array()
   array_of_ptrs = ( LP_c_double * rows )( ) #  LP_c_double_Array 
   
   for i in range( rows ):
      array_of_ptrs[ i ] = array_2d[ i ]
   
   
   ptr_to_pointer = LP_LP_c_double( array_of_ptrs )
     
   return ptr_to_pointer



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
      #row_pointers[i] = \
      #     cast(byref(data, i * cols * sizeof(c_double)), POINTER(c_double))

      # Ok
      row_pointers[i] = \
            (c_double * cols ).from_buffer( data, i*( cols * sizeof( c_double ) ) ) 


   return row_pointers



# void
def Src3() :

   # x
   nbinsx = 64
   # y
   nbinsy = 64

   # # #
   # x
   #global source, dest
   ## source = std.vector[ "Double_t *" ]( nbinsx ) # error
   ## dest   = std.vector[ "Double_t *" ]( nbinsx ) # error
   # instead
   #source = std.vector[ "Double_t *" ]( )
   #dest   = std.vector[ "Double_t *" ]( )

   ##for (Int_t i = 0; i < nbinsx; i++) {
   #for i in range(0, nbinsx, 1):
   #   # y
   #   source_i  =  [ Double_t() for _ in range( nbinsy ) ] # new
   #   dest_i    =  [ Double_t() for _ in range( nbinsy ) ] # new
   #   #
   #   source_i  = array( "d", source_i )
   #   dest_i    = array( "d", dest_i )
   #   #
   #   source.push_back( source_i  )
   #   dest.push_back( dest_i  )
   #
   # ... Too much complication. Let's keep it simple:
   #
   # [ x ] [ y ]
   global source, dest
   source = [ [ Double_t() for _ in range( nbinsy ) ] for _ in range( nbinsx ) ] 
   dest   = [ [ Double_t() for _ in range( nbinsy ) ] for _ in range( nbinsx ) ] 


      
   Dir   = gROOT.GetTutorialDir() # TString
   file  = Dir + TString( "/spectrum/TSpectrum2.root" ) # TString
   global f
   f = TFile.Open( file.Data() ) # TFile *

   gStyle.SetOptStat(0)

   global search
   search = f.Get("search1") # auto # (TH2F *)


   # # #
   global s
   s = TSpectrum2()


   #for (Int_t i = 0; i < nbinsx; i++) {
   for i in range(0, nbinsx, 1):
      #      for (Int_t j = 0; j < nbinsy; j++) {
      for j in range(0, nbinsy, 1):
         # Error
         #source[i][j] = Int_t( search.GetBinContent(i + 1, j + 1) )
         # Ok
         source[i][j] = Long_t( search.GetBinContent(i + 1, j + 1) )
         


   global source_ptr_ptr, dest_ptr_ptr
   source_ptr_ptr = to_c_double_ptr_ptr_FLAT( source )
   dest_ptr_ptr   = to_c_double_ptr_ptr_FLAT( dest )



   # Note:
   #     Alternatives for the type double**.
   #     None of the next below have been sucessfully received by the
   #                          # Error:
   #                          source.data(),
   #                          dest.data(),
   #                          source.ctypes.data,
   #                          dest.ctypes.data,
   #                          source,
   #                          dest,
   #                          source.ctypes.data_as(POINTER(c_double)),
   #                          dest.ctypes.data_as(POINTER(c_double)),
   #     TSpectrum.SearchHighRes method worked out well.
   #     An alternative way was to use row_pointers method of ctypes:
   #        row_pointers = (c_double * (rows*cols) ) ()
   #     which can be loaded by doing:
   #        row_pointers[ i ] = cast( byref( data, i * cols * sizeof( c_double ), POINTER( c_double ) ) )
   #     or:
   #        row_pointers[ i ] = (c_double * cols)().from_buffer( data, i * cols * sizeof( c_double ) )
   #     
   #     Besides, the method of std.vector[ "double*" ].data()
   #     Generates Error. It has to be Pythonized properly. The definition "def to_c_double_ptr_ptr_FLAT" 
   #     could be a start for such a pythonization. 


   # Ok:
   global nfound
   # Int_t
   nfound = s.SearchHighRes(
                            source_ptr_ptr,
                            dest_ptr_ptr,
                            nbinsx,
                            nbinsy,
                            2,
                            2,
                            False,
                            3,
                            False,
                            1,
                            )
   #   
   print( "Found %d candidate peaks\n" % nfound )
   #
   search.Draw("CONT")


   # # #
   global m
   m = TMarker()
   m.SetMarkerStyle(23)
   m.SetMarkerColor(kRed)
   #
   global PositionX, PositionY
   PositionX = s.GetPositionX() # Double_t *
   PositionY = s.GetPositionY() # Double_t *
   #
   #for (Int_t i = 0; i < nfound; i++) {
   for i in range(0, nfound, 1):
      print( "posx= %d, posy= %d, value=%d\n" % (
                int( PositionX[i] + 0.5 )                 ,
                int( PositionY[i] + 0.5 )                 ,
                int( source[                           \
                            int( PositionX[i] + 0.5 )  \
                            ]                          \
                           [                           \
                            int( PositionY[i] + 0.5 )  \
                            ]                          \
                     )                                    ,
                )
            ) 
      m.DrawMarker( PositionX[i], PositionY[i] )
      

  
   # Note: ctypes has a fixed way to define arrays. So we cannot clear 
   #       memory item by item. Instead we will use a simple garbage 
   #       collector of Python-style: del source_ptr_ptr
   #
   # Clean-up. 
   del source_ptr_ptr
   del dest_ptr_ptr

   # Clean-up. 
   while len(source) > 0 and len(dest) > 0: 
      del source[0]
      del dest[0]
      
   


if __name__ == "__main__":
   Src3()
