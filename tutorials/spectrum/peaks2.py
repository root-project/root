## \file
## \ingroup tutorial_spectrum
## \notebook
##
## Example to illustrate the 2-dim peak finder: class TSpectrum2.
##
## This script generates a random number of 2-dim gaussian peaks.
## Afterwards, the position of the peaks is found via TSpectrum2 methods.
## To execute this example, do:
##
## ~~~{.py}
##  IPython[1]: %run peaks2.py  # generates up to 50 peaks by default
##  IPython[2]: peaks2(10)      # generates up to 10 peaks
## ~~~
##
## The script will iterate itsself, by generating a new histogram, each time,
## having between 5 and the maximun number of peaks specified.
## Do a Double-Click on the bottom right corner of the pad, to go to a new spectrum.
## To Quit, select the "quit" item in the canvas "File" menu.
##
## \macro_image
## \macro_code
##
## \author Rene Brun
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
                   TF2,
                   TH2,
                   TH2F,
                   TMath,
                   TROOT,
                   TRandom,
                   TSpectrum2,
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

from ctypes import c_double

# utils
def to_c( ls ):
   return (c_double * len(ls) )( * ls )
def to_py( c_ls ):
   return list( c_ls ) 

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
s       = TSpectrum2()
h2      = TH2F() # nullptr
npeaks  = 30

# Double_t
def fpeaks2(x : float, par : float) :

   result = 0.1
   #   for (Int_t p = 0; p < npeaks; p++) {
   for p in range(0, npeaks, 1):

      norm    = par[5 * p + 0]
      mean1   = par[5 * p + 1]
      sigma1  = par[5 * p + 2]
      mean2   = par[5 * p + 3]
      sigma2  = par[5 * p + 4]

      result += \
                norm *\
                TMath.Gaus(x[0], mean1, sigma1) *\
                TMath.Gaus(x[1], mean2, sigma2)
      
   return result


   
# void
def findPeak2() :

   print("Generating histogram with %d peaks\n" % npeaks)
   
   # Setting-up ...
   nbinsx = 200
   nbinsy = 200
   #
   xmin = 0
   xmax = float( nbinsx )
   #
   ymin = 0
   ymax = float( nbinsy )
   #
   dx = (xmax - xmin) / nbinsx
   dy = (ymax - ymin) / nbinsy
   
   
   global h2
   if "h2" in globals() : 
      del h2
   #
   h2 = TH2F("h2", "test", nbinsx, xmin, xmax, nbinsy, ymin, ymax)
   h2.SetStats( False )

   
   # Generate n peaks at random.
   par = [ float() for _ in range(3000) ]
   #
   #   for (p = 0; p < npeaks; p++) {
   for p in range(0, npeaks, 1):
      par[5 * p + 0] = gRandom.Uniform( 0.2  , 1      )
      par[5 * p + 1] = gRandom.Uniform( xmin , xmax   )
      par[5 * p + 2] = gRandom.Uniform( dx   , 5 * dx )
      par[5 * p + 3] = gRandom.Uniform( ymin , ymax   )
      par[5 * p + 4] = gRandom.Uniform( dy   , 5 * dy )
      

   global f2
   f2 = TF2("f2", fpeaks2, xmin, xmax, ymin, ymax, 5 * npeaks)
   f2.SetNpx(100)
   f2.SetNpy(100)
   #
   c_par = to_c( par )
   f2.SetParameters( c_par )
   par = to_py( c_par )


   global c1
   c1 = gROOT.GetListOfCanvases().FindObject( "c1" ) # (TCanvas*)
   if (not c1) :
      c1 = TCanvas( "c1", "c1", 10, 10, 1000, 700 )

   h2.FillRandom( "f2", 500000 )
   
   # Now the real stuff: Finding the peaks.
   global nfound
   nfound = s.Search( h2, 2, "col" )
   
   # Searching good and ghost peaks (approximation).
   pf, ngood = int(0), int(0)
   global xpeaks, ypeaks
   xpeaks = s.GetPositionX()
   ypeaks = s.GetPositionY()

   #   for (p = 0; p < npeaks; p++) {
   #      for (pf = 0; pf < nfound; pf++) {
   for p in range(0, npeaks, 1):
      for pf in range(0, nfound, 1):

         diffx = TMath.Abs( xpeaks[pf] - par[5 * p + 1] )
         diffy = TMath.Abs( ypeaks[pf] - par[5 * p + 3] )

         if (diffx < 2 * dx and diffy < 2 * dy) :
            ngood += 1 
      
   if (ngood > nfound) :
      ngood = nfound


   # Search ghost peaks (by approximation).
   nghost = 0
   #   for (pf = 0; pf < nfound; pf++) {
   for pf in range(0, nfound, 1):
      nf = 0
      #      for (p = 0; p < npeaks; p++) {
      for p in range(0, npeaks, 1):
         diffx = TMath.Abs( xpeaks[pf] - par[5 * p + 1] )
         diffy = TMath.Abs( ypeaks[pf] - par[5 * p + 3] )

         if (diffx < 2 * dx and diffy < 2 * dy) :
            nf += 1
         
      if (nf == 0) :
         nghost += 1
      
   c1.Update()
   

   s.Print()
   print(
          "Gener=%d, Found=%d, Good=%d, Ghost=%d\n" % ( 
                                                       npeaks ,
                                                       nfound ,
                                                       ngood  ,
                                                       nghost ,
                                                       )
          )


   if not gROOT.IsBatch():
      print("\nDouble click in the bottom right corner of the pad to continue\n")
      c1.WaitPrimitive()
      
   
# void
def peaks2(maxpeaks : int = 50) :

   global s
   s = TSpectrum2( 2 * maxpeaks )

   #   for (int i = 0; i < 10; ++i) {
   for i in range(0, 10, 1):
   #for i in range(0, 2, 1):
      global npeaks
      npeaks = int( gRandom.Uniform( 5, maxpeaks ) ) 
      findPeak2()
      
   


if __name__ == "__main__":
   # peaks2() # 50 peaks by default
   peaks2(10)
   # peaks2(1)
