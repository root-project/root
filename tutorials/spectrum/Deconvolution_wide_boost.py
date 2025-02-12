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
def Deconvolution_wide_boost() :

   nbins = 256 
   
   xmin = 0.
   xmax = float( nbins )
   
   source   = [ float() for _ in range( nbins ) ]
   response = [ float() for _ in range( nbins ) ]


   gROOT.ForceStyle()

   
   #
   global h, d
   h = TH1F ( "h" , "Deconvolution" , nbins , xmin , xmax ) 
   d = TH1F ( "d" , ""              , nbins , xmin , xmax ) 
   

   Dir  = gROOT.GetTutorialDir()                       # TString
   file = Dir + TString( "/spectrum/TSpectrum.root" )  # TString
   global f
   f = TFile( file.Data() ) # TFile


   #
   h = f.Get( "decon3" ) # (TH1F *)
   h.SetTitle( 
               "Deconvolution of closely positioned overlapping"
               "peaks using boosted Gold deconvolution method"
               )
   

   #
   d = f.Get( "decon_response_wide" ) # (TH1F *)

   
   # - - Get Bin -- #
   #
   #for (i = 0; i < nbins; i++) {
   for i in range(0, nbins, 1):
      source[i] = h.GetBinContent( i + 1 )
   #
   #for (i = 0; i < nbins; i++) {
   for i in range(0, nbins, 1):
      response[i] = d.GetBinContent( i + 1 )
      

   h.SetMaximum ( 200000 )
   h.Draw       ( "L"    )



   # # #
   c_source   = to_c ( source   )
   c_response = to_c ( response )
   #
   global s
   s = TSpectrum() # TSpectrum

   s.Deconvolution(
                    c_source   ,   # source            : Double_t*
                    c_response ,   # response          : Double_t*
                    256        ,   # ssize             : Int_t
                    200        ,   # numberIterations  : Int_t
                    50         ,   # numberRepetitions : Int_t
                    1.2        ,   # boost             : Double_t
                    )


   
   #for (i = 0; i < nbins; i++) {
   for i in range(0, nbins, 1):
      d.SetBinContent( i + 1, c_source[i] )

 
   d.SetLineColor ( kRed     )
   d.Draw         ( "SAME L" )
   


if __name__ == "__main__":
   Deconvolution_wide_boost()
