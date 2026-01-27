## \file
## \ingroup tutorial_spectrum
## \notebook
##
## Example to illustrate smoothing using Markov algorithm (class TSpectrum).
##
## \macro_image
## \macro_code
##
## \author Miroslav Morhac
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
                   TSpectrum,
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

from ctypes import c_double

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



# void
def Smoothing() :

   nbins = 1024 
   
   xmin = 0.  
   xmax = float( nbins )
   
   source = [ float() for _ in range( nbins ) ]


   gROOT.ForceStyle()


   Dir  = gROOT.GetTutorialDir()                       # TString
   file = Dir + TString( "/spectrum/TSpectrum.root" )  # TString
   global f
   f = TFile( file.Data() ) # TFile


   global h
   h = f.Get( "back1" ) # (TH1F *)
   h.SetTitle( "Smoothed spectrum for m=3" )

   
   # Putting bins onto source.
   #
   #for (i = 0; i < nbins; i++) {
   for i in range(0, nbins, 1):
      source[i] = h.GetBinContent( i + 1 )

   h.SetAxisRange( 1, 1024 )
   h.Draw( "L" )
   


   # # #
   global s
   s = TSpectrum() # TSpectrum

   
   global c_source
   c_source = to_c( source )


   # # # 
   s.SmoothMarkov(
                   c_source ,   # source     : Double_t*
                   1024     ,   # ssize      : Int_t
                   3        ,   # averWindow : Int_t  

                   ) # -> const char*

   # Note:
   #       Tested on 3 or 7 or 10 for "averWindow" argument.


   global smooth
   smooth = TH1F( "smooth", "smooth", nbins, 0., nbins )
   smooth.SetLineColor( kRed )

 
   #for (i = 0; i < nbins; i++) {
   for i in range(0, nbins, 1):
      smooth.SetBinContent( i + 1, c_source[i] )
   # #
   smooth.Draw( "L SAME" )
   


if __name__ == "__main__":
   Smoothing()
