## \file
## \ingroup tutorial_spectrum
## \notebook
##
## Example to illustrate the background estimator (class TSpectrum).
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



# void
def Background_decr() :

   nbins = 1024
   
   xmin = 0.
   xmax = float( nbins )
   
   source = [ float() for _ in range( nbins ) ]


   gROOT.ForceStyle()
   

   global d
   d = TH1F ( "d", "", nbins, xmin, xmax )

   
   Dir  = gROOT.GetTutorialDir()                       # TString
   file = Dir + TString( "/spectrum/TSpectrum.root" )  # TString
   global f
   f = TFile( file.Data() ) # TFile


   global back
   back = f.Get( "back1" ) # (TH1F *)
   back.SetTitle( "Estimation of background with decreasing window" )

   back.GetXaxis().SetRange( 1, nbins )
   back.Draw( "L" )
   

   # # #
   global s
   s = TSpectrum() # TSpectrum


   
   #for (i = 0; i < nbins; i++) {
   for i in range(0, nbins, 1):
      source[i] = back.GetBinContent(i + 1)
      
   
   # to c types
   global c_source
   c_source = to_c( source )

   # # #
   # Estimate the background
   #
   s.Background(
                 c_source                        ,    # ssize
                 nbins                           ,    # spectrum
                 6                               ,    # smoothing
                 TSpectrum.kBackDecreasingWindow ,    # smoothWindow
                 TSpectrum.kBackOrder2           ,    # numberIterations
                 False                           ,    # filterOrder
                 TSpectrum.kBackSmoothing3       ,    # direction
                 False                           ,    # compton
                 ) # const char* 

   

   # # #
   # Draw the estimated background
   #
   #for (i = 0; i < nbins; i++) {
   for i in range(0, nbins, 1):
      d.SetBinContent( i + 1, c_source[i] )
   #
   d.SetLineColor ( kRed     )
   d.Draw         ( "SAME L" )
   


if __name__ == "__main__":
   Background_decr()
