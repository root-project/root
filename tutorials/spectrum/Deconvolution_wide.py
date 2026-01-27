## \file
## \ingroup tutorial_spectrum
## \notebook
##
## Example to illustrate deconvolution function (class TSpectrum).
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
def Deconvolution_wide() :

   nbins = 256
   
   xmin = 0.
   xmax = float( nbins )
   
   source   = [ float() for _ in range( nbins ) ]
   response = [ float() for _ in range( nbins ) ]

   
   gROOT.ForceStyle()


   Dir  = gROOT.GetTutorialDir()                       # TString
   file = Dir + TString( "/spectrum/TSpectrum.root" )  # TString
   global f
   f = TFile( file.Data() ) # TFile


   global h, d
   h = f.Get ( "decon3"              ) # (TH1F *)
   d = f.Get ( "decon_response_wide" ) # (TH1F *)
   

   #for (i = 0; i < nbins; i++) {
   for i in range(0, nbins, 1):
      source[i] = h.GetBinContent( i + 1 )
      
   #for (i = 0; i < nbins; i++) {
   for i in range(0, nbins, 1):
      response[i] = d.GetBinContent( i + 1 )
      
   
   h.SetTitle(
               "Deconvolution of closely positioned "
               "overlapping peaks using Gold deconvolution "
               "method."
               )

   h.SetMaximum ( 50000 )
   h.Draw       ( "L"   )


   # # #
   global s
   s = TSpectrum() # TSpectrum


   #to_c
   global c_source, c_response
   c_source   = to_c( source   )
   c_response = to_c( response )

   
   s.Deconvolution(
                    c_source   ,   # Input signal data.
                    c_response ,   # Instrument response function.
                    256        ,   # Output histogram bins.
                    10000      ,   # numberRepetitions.
                    1          ,   # numberIterations.
                    1          ,   # boost.
                    )

   
   #for (i = 0; i < nbins; i++) {
   for i in range(0, nbins, 1):
      d.SetBinContent( i + 1, c_source[i] )
      

   d.SetLineColor ( kRed     )
   d.Draw         ( "SAME L" )
   


if __name__ == "__main__":
   Deconvolution_wide()
