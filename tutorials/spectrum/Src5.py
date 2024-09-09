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


#standard library
std = ROOT.std
make_shared = std.make_shared
unique_ptr = std.unique_ptr

#classes
TSpectrum2 = ROOT.TSpectrum2 
TFile = ROOT.TFile
TMarker = ROOT.TMarker
TString = ROOT.TString
TCanvas = ROOT.TCanvas
TH1F = ROOT.TH1F
TGraph = ROOT.TGraph
TLatex = ROOT.TLatex

#maths
sin = ROOT.sin
cos = ROOT.cos
sqrt = ROOT.sqrt

#types
Double_t = ROOT.Double_t
Bool_t = ROOT.Bool_t
Float_t = ROOT.Float_t
Int_t = ROOT.Int_t
nullptr = ROOT.nullptr
c_double = ctypes.c_double
POINTER = ctypes.POINTER
sizeof = ctypes.sizeof

#utils
def to_c( ls ):
   return (c_double * len(ls) )( * ls )
def printf(string, *args):
   print( string % args, end="")
def sprintf(buffer, string, *args):
   buffer = string % args 
   return buffer

#constants
kBlue = ROOT.kBlue
kRed = ROOT.kRed
kGreen = ROOT.kGreen

#globals
gStyle = ROOT.gStyle
gPad = ROOT.gPad
gRandom = ROOT.gRandom
gBenchmark = ROOT.gBenchmark
gROOT = ROOT.gROOT



# For C++ type: double **
def to_c_double_ptr_ptr_FLAT( matrix ) :

   rows, cols = len(matrix), len(matrix[0])

   data = ( c_double * ( rows * cols ) )()
   row_pointers = ( POINTER( c_double ) * rows )()
   
   # fill data
   for i in range(rows) :
      for j in range(cols) :
         data[ i*cols + j ] = matrix[ i ][ j ] 
  
   # fill addresses
   for i in range( rows ) :
      # Ok
      #row_pointers[i] = cast(byref(data, i * cols * sizeof(c_double)), POINTER(c_double))

      # Ok
      row_pointers[i] = (c_double * cols ).from_buffer( data, i*( cols * sizeof( c_double) ) ) 


   return row_pointers



# void
def Src5() :

   nbinsx = 64 # Int_t
   nbinsy = 64 # Int_t


   ##std.vector<Double_t *> source(nbinsx), dest(nbinsx)
   # 
   ##for (Int_t i = 0; i < nbinsx; i++) {
   #for i in range(0, nbinsx, 1):
   #   source[i] =  Double_t[nbinsy]; # new
   #   dest[i]   =  Double_t[nbinsy]; # new
   #
   source = [ [ Double_t() for _ in range( nbinsy ) ] for _ in range( nbinsx ) ] 
   dest   = [ [ Double_t() for _ in range( nbinsy ) ] for _ in range( nbinsx ) ]
      
   Dir = gROOT.GetTutorialDir(); # TString
   file = Dir + TString( "/spectrum/TSpectrum2.root" ); # TString
   global f
   f = TFile.Open(file.Data()) # TFile *

   gStyle.SetOptStat(0)

   global search
   search = f.Get("search3;1"); # auto # (TH2F *)


   # # #
   global s
   s = TSpectrum2()
   #for (Int_t i = 0; i < nbinsx; i++) {
   for i in range(0, nbinsx, 1):
      #      for (Int_t j = 0; j < nbinsy; j++) {
      for j in range(0, nbinsy, 1):
         source[i][j] = search.GetBinContent(i + 1, j + 1)
         
      
   # double** type
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
                            False,
                            10,
                            False,
                            1,
   )
   #
   printf("Found %d candidate peaks\n", nfound)
   
   global PositionX, PositionY
   PositionX = s.GetPositionX() # Double_t *
   PositionY = s.GetPositionY() # Double_t *

   search.Draw("COL")

   global m
   m = TMarker()
   m.SetMarkerStyle(23)
   m.SetMarkerColor(kRed)

   #for (Int_t i = 0; i < nfound; i++) {
   for i in range(0, nfound, 1):
      printf("posx= %d, posy= %d, value=%d\n",
             Int_t( PositionX[i] + 0.5 ) ,
             Int_t( PositionY[i] + 0.5 ) ,
             Int_t( source[Int_t( PositionX[i] + 0.5 ) ][Int_t( PositionY[i] + 0.5 ) ] ),
            )
      m.DrawMarker(PositionX[i], PositionY[i])
      
   
   ##for (Int_t i = 0; i < nbinsx; i++) {
   #for i in range(0, nbinsx, 1):
   #   del source[i]
   #   del dest[i]
   del source
   del source_ptr_ptr
   del dest_ptr_ptr
   del dest
      
   


if __name__ == "__main__":
   Src5()
