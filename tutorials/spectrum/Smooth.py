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


#standard library
std = ROOT.std
make_shared = std.make_shared
unique_ptr = std.unique_ptr

#classes
TSpectrum2 = ROOT.TSpectrum2 
TFile = ROOT.TFile
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
byref = ctypes.byref

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
      row_pointers[i] = (c_double * cols ).from_buffer( data, i*( cols * sizeof( c_double ) ) ) 


   return row_pointers



# void
def Smooth() :

   #
   i = Int_t()
   #
   # Error:
   #nbinsx = 256 # Int_t
   #nbinsy = 256 # Int_t
   # Error:
   #nbinsx = 200 # Int_t
   #nbinsy = 200 # Int_t
   # Ok
   nbinsx = 150 # Int_t
   nbinsy = 150 # Int_t
   # Ok
   #nbinsx = 100 # Int_t
   #nbinsy = 100 # Int_t
   #
   xmin = 0; # Double_t
   xmax = Double_t( nbinsx ) #  # Double_t
   #
   ymin = 0; # Double_t
   ymax = Double_t( nbinsy ) #  # Double_t
   
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
   source = [ [ Double_t() for _ in range( nbinsy ) ] for _ in range( nbinsx ) ] 

      
   Dir = gROOT.GetTutorialDir(); # TString
   file = Dir + TString( "/spectrum/TSpectrum2.root" ) ; # TString
   global f
   f = TFile(file.Data()); # TFile

   global smooth
   smooth = f.Get("smooth1"); # auto # (TH2F *)

   gStyle.SetOptStat(0)


   # # #
   global s
   s = TSpectrum2(); # auto


   #for (i = 0; i < nbinsx; i++) {
   for i in range(0, nbinsx, 1):
      #      for (j = 0; j < nbinsy; j++) {
      for j in range(0, nbinsy, 1):
         source[i][j] = smooth.GetBinContent(i + 1, j + 1)
         

   # to double** type
   global source_ptr_ptr
   source_ptr_ptr   = to_c_double_ptr_ptr_FLAT( source )
      
   # # #
   s.SmoothMarkov(
                  source_ptr_ptr,
                  nbinsx,
                  nbinsx,

                  # 1 or 3 or 5 or 7
                  3, 
   )



   #for (i = 0; i < nbinsx; i++) {
   for i in range(0, nbinsx, 1):
      #      for (j = 0; j < nbinsy; j++) {
      for j in range(0, nbinsy, 1):
         smooth.SetBinContent(i + 1, j + 1, source_ptr_ptr[i][j])
         
      
   smooth.Draw("SURF2")
   


if __name__ == "__main__":
   Smooth()
