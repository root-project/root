## \file
## \ingroup tutorial_spectrum
## \notebook
##
## Example to illustrate the background estimator of 
## the TSpectrum class including Comptong edges.
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
                   TMarker,
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



# void
def Background_smooth() :

   nbins = 4096
   xmin  = 0.
   xmax  = float( nbins )

   global source
   source = [ float() for _ in range( nbins ) ]
   source = array( "d", source ) 

   gROOT.ForceStyle()
   
   global d1, d2
   d1 = TH1F ( "d1", "", nbins, xmin, xmax )
   d2 = TH1F ( "d2", "", nbins, xmin, xmax )
   
   Dir  = gROOT.GetTutorialDir()                       # TString
   file = Dir + TString( "/spectrum/TSpectrum.root" )  # TString
   global f
   f = TFile( file.Data() ) # TFile

   global back
   back = f.Get("back1") # (TH1F *)
   back.SetTitle("Estimation of background with noise")
   back.SetAxisRange(3460, 3830)
   back.Draw("L")
   


   # # #
   global s
   s = TSpectrum()


   # - - d1 - - # 
   #for (i = 0; i < nbins; i++) {
   for i in range(0, nbins, 1):
      source[i] = back.GetBinContent( i + 1 )
   
   s.Background(
                 source                          ,
                 nbins                           ,
                 6                               ,
                 TSpectrum.kBackDecreasingWindow ,
                 TSpectrum.kBackOrder2           ,
                 False                           ,
                 TSpectrum.kBackSmoothing3       ,
                 False                           ,
                 )
  
   
   #for (i = 0; i < nbins; i++) {
   for i in range(0, nbins, 1):
      d1.SetBinContent( i + 1, source[i] )
   
   d1.SetLineColor ( kRed     )
   d1.Draw         ( "SAME L" )

   

   # - - d2 - - # 
   #for (i = 0; i < nbins; i++) {
   for i in range(0, nbins, 1):
      source[i] = back.GetBinContent( i + 1 )
   
   s.Background(
                 source                          ,
                 nbins                           ,
                 6                               ,
                 TSpectrum.kBackDecreasingWindow ,
                 TSpectrum.kBackOrder2           ,
                 True                            ,
                 TSpectrum.kBackSmoothing3       ,
                 False                           ,
                 )
   
  
   #for (i = 0; i < nbins; i++) {
   for i in range(0, nbins, 1):
      d2.SetBinContent( i + 1, source[i] )
   
   d2.SetLineColor ( kBlue    )
   d2.Draw         ( "SAME L" )
   


if __name__ == "__main__":
   Background_smooth()
