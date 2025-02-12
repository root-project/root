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
def Deconvolution() :

   nbins = 256 
   
   xmin = float( 0     ) 
   xmax = float( nbins ) 
   
   source   = [ float() for _ in range( nbins ) ]
   response = [ float() for _ in range( nbins ) ]


   gROOT.ForceStyle()
   

   Dir  = gROOT.GetTutorialDir()                       # TString
   file = Dir + TString( "/spectrum/TSpectrum.root" )  # TString
   global f
   f = TFile( file.Data() ) # TFile

   global h, d
   h = f.Get ( "decon1"         )   # (TH1F *)
   d = f.Get ( "decon_response" )   # (TH1F *)
   

   #for (i = 0; i < nbins; i++) {
   for i in range(0, nbins, 1):
      source[i] = h.GetBinContent( i + 1 )
      
   #for (i = 0; i < nbins; i++) {
   for i in range(0, nbins, 1):
      response[i] = d.GetBinContent( i + 1 )
      
   
   h.SetTitle   ( "Deconvolution" )
   h.SetMaximum ( 30000           )
   h.Draw       ( "L"             )


   # # #
   global s
   s = TSpectrum() # TSpectrum

   # Handling types. 
   c_source = to_c   ( source   )
   c_response = to_c ( response )

   # # #
   s.Deconvolution(
                    c_source   ,   # source            : Double_t*
                    c_response ,   # response          : Double_t*
                    256        ,   # ssize             : Int_t
                    1000       ,   # numberIterations  : Int_t
                    1          ,   # numberRepetitions : Int_t
                    1.         ,   # boost             : Double_t
                    ) 

   
   #for (i = 0; i < nbins; i++) {
   for i in range(0, nbins, 1):
      d.SetBinContent( i + 1, c_source[i] )

   
   d.SetLineColor ( kRed     )
   d.Draw         ( "SAME L" )
   


if __name__ == "__main__":
   Deconvolution()
