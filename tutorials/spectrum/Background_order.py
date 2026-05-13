## \file
## \ingroup tutorial_spectrum
## \notebook
##
## Example illustrating the influence of the clipping-filter-difference-order
## on some estimated background.
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
                   TFile,
                   TSpectrum,
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

# ctypes
from ctypes import (
                     c_double,
                     )

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
      #

                   gStyle,
                   gPad,
                   gRandom,
                   gBenchmark,
                   gROOT,
                   )


# void
def Background_order() :

   nbins  = 4096 
   xmin   = 0. 
   xmax   = 4096. 

   global source, dest
   source = [ float() for _ in range( nbins ) ]
   dest   = [ float() for _ in range( nbins ) ]
   source = array ( "d", source )
   dest   = array ( "d", dest   )

   gROOT.ForceStyle()
   
   global d1, d2, d3, d4
   d1 = TH1F ( "d1", "", nbins, xmin, xmax ) # TH1F
   d2 = TH1F ( "d2", "", nbins, xmin, xmax ) # TH1F
   d3 = TH1F ( "d3", "", nbins, xmin, xmax ) # TH1F
   d4 = TH1F ( "d4", "", nbins, xmin, xmax ) # TH1F
   
   Dir  = gROOT.GetTutorialDir()                       # TString
   file = Dir + TString( "/spectrum/TSpectrum.root" )  # TString
   global f
   f = TFile( file.Data() ) # TFile

   global back
   back = f.Get( "back2" ) # (TH1F *)
   back.SetTitle(
                  "Influence of clipping filter difference "
                  "order on the estimated background."
                  )
   back.SetAxisRange ( 1220, 1460 )
   back.SetMaximum   ( 3000       )
   back.Draw         ( "L"        )
   


   global s
   s = TSpectrum()
   
   #for (i = 0; i < nbins; i++) {
   for i in range(0, nbins, 1):
      source[i] = back.GetBinContent( i + 1 )
   # #  
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
      d1.SetBinContent( i + 1, source[i] )
   # #   
   d1.SetLineColor ( kRed     )
   d1.Draw         ( "SAME L" )
   
   #for (i = 0; i < nbins; i++) {
   for i in range(0, nbins, 1):
      source[i] = back.GetBinContent( i + 1 )
   # #    
   s.Background(
                source                          ,
                nbins                           ,
                40                              ,
                TSpectrum.kBackDecreasingWindow ,
                TSpectrum.kBackOrder4           ,
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
                nbins                           ,
                40                              ,
                TSpectrum.kBackDecreasingWindow ,
                TSpectrum.kBackOrder6           ,
                False                           ,
                TSpectrum.kBackSmoothing3       ,
                False                           ,
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
                source,
                nbins,
                40,
                TSpectrum.kBackDecreasingWindow,
                TSpectrum.kBackOrder8,
                False,
                TSpectrum.kBackSmoothing3,
                False,
                )


   #for (i = 0; i < nbins; i++) {
   for i in range(0, nbins, 1):
      d4.SetBinContent( i + 1, source[i] )
   #
   d4.SetLineColor ( kMagenta )
   d4.Draw         ( "SAME L" )
   

   gROOT.Remove( s ) #  s is TSpectrum

if __name__ == "__main__":
   Background_order()
