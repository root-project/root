## \file
## \ingroup tutorial_spectrum
## \notebook
##
## Example to illustrate a high resolution peak searching function 
## using the TSpetrum class.
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
                   TString,
                   TFile,
                   TSpectrum,
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

from ctypes import c_double

# utils
def to_c( ls ):
   return ( c_double * len( ls ) )( * ls )

# constants
from ROOT import (
                   kOrange,
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
def SearchHR1() :

   fPositionX = [ float() for _ in range(100) ]
   fPositionY = [ float() for _ in range(100) ]
   fPositionX = array( "d", fPositionX )
   fPositionY = array( "d", fPositionY )
 
   fNPeaks = 0


   nbins = 1024
   xmin  = 0.
   xmax  = float( nbins )
   a     = float()

  
   source = [ float() for _ in range( nbins ) ]
   dest   = [ float() for _ in range( nbins ) ]
   source = array( "d", source )
   dest   = array( "d", dest   )
   
   gROOT.ForceStyle()

   Dir  = gROOT.GetTutorialDir()                       # TString
   file = Dir + TString( "/spectrum/TSpectrum.root" )  # TString
   global f
   f = TFile( file.Data() ) # TFile

   global h
   h = f.Get( "back2" ) # (TH1F *)
   h.SetTitle( "High resolution peak searching, number of iterations = 3" )
   h.GetXaxis().SetRange( 1, nbins )

   global d
   d = TH1F( "d", "", nbins, xmin, xmax ) # TH1F
   h.Draw( "L" )
   
   #for (i = 0; i < nbins; i++) {
   for i in range(0, nbins, 1):
      source[i] = h.GetBinContent( i + 1 )
   
   h.Draw( "L" )
   

   
   global s
   s = TSpectrum()
   
   global nfound
   nfound = s.SearchHighRes(
                             source,
                             dest,
                             nbins,
                             8,
                             2,
                             True,
                             3,
                             True,
                             3,
                             )
   global xpeaks
   xpeaks = s.GetPositionX() # Double_t *
   #
   #for (i = 0; i < nfound; i++) {
   for i in range(0, nfound, 1):
      a             = xpeaks[i]
      Bin           = 1 + int( a + 0.5 )
      
      fPositionX[i] = h.GetBinCenter  ( Bin )
      fPositionY[i] = h.GetBinContent ( Bin )
      
   
   global pm
   pm = h.GetListOfFunctions().FindObject( "TPolyMarker" )  # (TPolyMarker *)
   if (pm)  :
      h.GetListOfFunctions().Remove( pm )
      del pm
   
   pm = TPolyMarker( nfound, fPositionX, fPositionY )
   h.GetListOfFunctions().Add( pm )
   
   pm.SetMarkerStyle ( 23   )
   pm.SetMarkerColor ( kRed )
   pm.SetMarkerSize  ( 1.3  )

   
   #for (i = 0; i < nbins; i++) {
   for i in range(0, nbins, 1):
      d.SetBinContent( i + 1, dest[i] )
   
   d.SetLineColor ( kRed   )
   d.Draw         ( "SAME" )
   

   
   print( "Found %d candidate peaks\n" % nfound )
   #for (i = 0; i < nfound; i++) {
   for i in range(0, nfound, 1):
      print( "posx= %f, posy= %f\n" % ( fPositionX[i], fPositionY[i] ) )
      
   


if __name__ == "__main__":
   SearchHR1()
