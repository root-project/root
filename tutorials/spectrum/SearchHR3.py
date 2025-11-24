## \file
## \ingroup tutorial_spectrum
## \notebook
##
## Example illustrating the influence of: Number of iterations of a deconvolution process
## in a high-resolution-peak-searching-function using the TSpectrum class.
##
## \macro_output
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
                   TString,
                   TFile,
                   TPolyMarker,
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
def SearchHR3() :

   fPositionX = [ float() for _ in range( 100 ) ]
   fPositionY = [ float() for _ in range( 100 ) ]

   fPositionX  = array( "d", fPositionX )
   fPositionY  = array( "d", fPositionY )

   fNPeaks     = 0

   nbins  = 1024 
   xmin   = 0. 
   xmax   = float( nbins ) 
   a      = float()
   i      = int()

   
   source = [ float() for _ in range( nbins ) ]
   dest   = [ float() for _ in range( nbins ) ]
   source = array( "d", source )
   dest   = array( "d", dest   )

   gROOT.ForceStyle()
   
   Dir   = gROOT.GetTutorialDir()                     # TString
   file  = Dir + TString("/spectrum/TSpectrum.root")  # TString
   global f
   f = TFile( file.Data() ) # TFile


   global h
   h = f.Get( "back2" ) # (TH1F *)
   h.SetTitle("Influence of # of iterations in deconvolution in peak searching.")
   h.GetXaxis().SetRange( 1, nbins )
   
   global d1, d2, d3, d4
   d1 = TH1F("d1", "", nbins, xmin, xmax) # TH1F
   d2 = TH1F("d2", "", nbins, xmin, xmax) # TH1F
   d3 = TH1F("d3", "", nbins, xmin, xmax) # TH1F
   d4 = TH1F("d4", "", nbins, xmin, xmax) # TH1F
   
   global s
   s = TSpectrum() # TSpectrum
   
   #for (i = 0; i < nbins; i++) {
   for i in range(0, nbins, 1):
      source[i] = h.GetBinContent(i + 1)
      

   global nfound, npeaks
   nfound = s.SearchHighRes( 
                             source ,
                             dest   ,
                             nbins  ,
                             8      ,
                             2      ,
                             True   ,
                             3      ,
                             True   ,
                             3      ,
                             )

   xpeaks = s.GetPositionX() # Double_t *
   
   #for (i = 0; i < nfound; i++) {
   for i in range(0, nfound, 1):
      #
      a             = xpeaks[i]
      Bin           = 1 + int(a + 0.5)
      #
      fPositionX[i] = h.GetBinCenter  ( Bin )
      fPositionY[i] = h.GetBinContent ( Bin )
      

   global pm
   pm = h.GetListOfFunctions().FindObject( "TPolyMarker" ) # (TPolyMarker *)
   if (pm)  :
      # Caution: "TList" will be deprecated in future ROOT versions > 6.
      h.GetListOfFunctions().Remove(pm) 
      del pm

      
   pm = TPolyMarker(
                     nfound     ,
                     fPositionX ,
                     fPositionY ,
                     ) # new

   h.GetListOfFunctions().Add( pm )
   #
   pm.SetMarkerStyle ( 23   )
   pm.SetMarkerColor ( kRed )
   pm.SetMarkerSize  ( 1.3  )

   
   h.Draw( "L" )
   

   #for (i = 0; i < nbins; i++) {
   for i in range(0, nbins, 1):
      d1.SetBinContent( i + 1, dest[i] )
      
   d1.SetLineColor ( kRed   )
   d1.Draw         ( "SAME" )
   

   #for (i = 0; i < nbins; i++) {
   for i in range(0, nbins, 1):
      source[i] = h.GetBinContent( i + 1 )
      
   s.SearchHighRes(
                    source ,
                    dest   ,
                    nbins  ,
                    8      ,
                    2      ,
                    True   ,
                    10     ,
                    True   ,
                    3      ,
                    )
   #for (i = 0; i < nbins; i++) {
   for i in range(0, nbins, 1):
      d2.SetBinContent( i + 1, dest[i] )
   # #   
   d2.SetLineColor ( kBlue  )
   d2.Draw         ( "SAME" )

   
   #for (i = 0; i < nbins; i++) {
   for i in range(0, nbins, 1):
      source[i] = h.GetBinContent( i + 1 )
      
   s.SearchHighRes(
                    source ,
                    dest   ,
                    nbins  ,
                    8      ,
                    2      ,
                    True   ,
                    100    ,
                    True   ,
                    3      ,
                    )

   #for (i = 0; i < nbins; i++) {
   for i in range(0, nbins, 1):
      d3.SetBinContent( i + 1, dest[i] )
   # #      
   d3.SetLineColor ( kGreen )
   d3.Draw         ( "SAME" )
   

   #for (i = 0; i < nbins; i++) {
   for i in range(0, nbins, 1):
      source[i] = h.GetBinContent( i + 1 )
      

   s.SearchHighRes(
                    source ,
                    dest   ,
                    nbins  ,
                    8      ,
                    2      ,
                    True   ,
                    1000   ,
                    True   ,
                    3      ,
                    )
   #for (i = 0; i < nbins; i++) {
   for i in range(0, nbins, 1):
      d4.SetBinContent( i + 1, dest[i] )
   # #   
   d4.SetLineColor ( kMagenta )
   d4.Draw         ( "SAME"   )
   

   print( "Found %d candidate peaks\n" % nfound )
   


if __name__ == "__main__":
   SearchHR3()
