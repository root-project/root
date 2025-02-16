## \file
## \ingroup tutorial_spectrum
## \notebook
##
## Example to illustrate the influence of the clipping window width on
## some estimated background.
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
                   TString,
                   TSpectrum,
                   TFile,
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
#
from ctypes import c_double

# utils
def to_c( ls ):
   return ( c_double * len( ls ) )( * ls )

# constants
from ROOT import (
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



# void
def Background_width2() :

   nbins = 4096
   xmin  = 0.
   xmax  = 4096.

   source = [ float() for _ in range( nbins ) ]
   source = array( "d", source )

   gROOT.ForceStyle()
   
   global d1, d2, d3, d4
   d1 = TH1F("d1", "", nbins, xmin, xmax) # TH1F
   d2 = TH1F("d2", "", nbins, xmin, xmax) # TH1F
   d3 = TH1F("d3", "", nbins, xmin, xmax) # TH1F
   d4 = TH1F("d4", "", nbins, xmin, xmax) # TH1F
   
   Dir  = gROOT.GetTutorialDir()                       # TString
   file = Dir + TString( "/spectrum/TSpectrum.root" )  # TString
   global f
   f = TFile( file.Data() ) # TFile

   global back
   back = f.Get( "back2" ) # (TH1F *)
   
   back.SetTitle(
                  "Influence of clipping window width"
                  " on the estimated background."
                  )
   back.SetAxisRange ( 0, 1000 )
   back.SetMaximum   ( 7000    )
   back.Draw         ( "L"     )

   
   # # #


   global s
   s = TSpectrum() # TSpectrum
   
   #for (i = 0; i < nbins; i++) {
   for i in range(0, nbins, 1):
      source[i] = back.GetBinContent( i + 1 )
   #
   s.Background(
                 source                          ,
                 nbins                           ,
                 10                              ,
                 TSpectrum.kBackDecreasingWindow ,
                 TSpectrum.kBackOrder2           ,
                 False                           ,
                 TSpectrum.kBackSmoothing3       ,
                 False                           ,
                 )
   #for (i = 0; i < nbins; i++) {
   for i in range(0, nbins, 1):
      d1.SetBinContent( i + 1, source[i] )
   # #    
   d1.SetLineColor ( kRed     )
   d1.Draw         ( "SAME L" )
   

   #for (i = 0; i < nbins; i++) {
   for i in range(0, nbins, 1):
      source[i] = back.GetBinContent( i + 1 )
   #
   s.Background(
                source                          ,
                nbins                           ,
                20                              ,
                TSpectrum.kBackDecreasingWindow ,
                TSpectrum.kBackOrder2           ,
                False                           ,
                TSpectrum.kBackSmoothing3       ,
                False                           ,
                )

   #for (i = 0; i < nbins; i++) {
   for i in range(0, nbins, 1):
      d2.SetBinContent( i + 1, source[i] )
   # #
   d2.SetLineColor ( kBlue    )
   d2.Draw         ( "SAME L" )
   

   #for (i = 0; i < nbins; i++) {
   for i in range(0, nbins, 1):
      source[i] = back.GetBinContent( i + 1 )
   # #      
   s.Background(
                 source                          ,
                 nbins,
                 30,
                 TSpectrum.kBackDecreasingWindow ,
                 TSpectrum.kBackOrder2           ,
                 False,
                 TSpectrum.kBackSmoothing3       ,
                 False,
                 )
   #for (i = 0; i < nbins; i++) {
   for i in range(0, nbins, 1):
      d3.SetBinContent( i + 1, source[i] )
   # #
   d3.SetLineColor ( kGreen   )
   d3.Draw         ( "SAME L" )
   

   #for (i = 0; i < nbins; i++) {
   for i in range(0, nbins, 1):
      source[i] = back.GetBinContent( i + 1 )
   #
   s.Background(
                 source                          ,
                 nbins                           ,
                 40                              ,
                 TSpectrum.kBackDecreasingWindow ,
                 TSpectrum.kBackOrder2           ,
                 False                           ,
                 TSpectrum.kBackSmoothing3       ,
                 False                           ,
                 )
   #for (i = 0; i < nbins; i++) {
   for i in range(0, nbins, 1):
      d4.SetBinContent( i + 1, source[i] )
   #
   d4.SetLineColor ( kMagenta )
   d4.Draw         ( "SAME L" )
   


if __name__ == "__main__":
   Background_width2()
