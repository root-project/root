## \file
## \ingroup tutorial_spectrum
## \notebook
##
## This script fits a spectrum(e.g. from a nuclear source) using 
## the Adaptive Weighted Mean Integrtion(AWMI) algorithm.
## 
## It uses the "TSpectrumFit" class to fit the data and 
## uses "TSpectrum" class to find peaks.
##
## To try this macro, in a ROOT (5 or 6 or 7) prompt, do:
##
## ~~~{.py}
##  IPython [0] : %run FitAwmi.py
##  IPython [1] : FitAwmi() # re-run with another random set of peaks
## ~~~
##
## \macro_image
## \macro_output
## \macro_code
##
## \author
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
                   TCanvas,
                   TF1,
                   TH1,
                   TH1F,
                   TList,
                   TMath,
                   TPolyMarker,
                   TROOT,
                   TRandom,
                   TSpectrum,
                   TSpectrumFit,
                   TH1F,
                   TGraph,
                   TLatex,
                   TPaveLabel,
                   TPavesText,
                   TPaveText,
                   TText,
                   TArrow,
                   TWbox,
                   TPad,
                   TBox,
                   TPad,
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
                   Float_t,
                   Bool_t,
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





# TH1F
def FitAwmi_Create_Spectrum( ) :

   nbins  =  1000
   xmin   = -10.
   xmax   =  10.

   # Prevent "memory leak".
   gROOT.Remove( gROOT.FindObject( "h" ) )

   h = TH1F( "h", "simulated spectrum", nbins, xmin, xmax )
   h.SetStats( False )

   f = TF1( "f", "TMath::Gaus(x, [0], [1], 1)", xmin, xmax )
   # f.SetParNames("mean", "sigma") 

   gRandom.SetSeed( 0 )  # Make it really random.

   # Create well separated peaks with exactly known means and areas.
   # Note: TSpectrumFit assumes that all peaks have the same sigma.
   global sigma
   sigma = (xmax - xmin) / float(nbins) * int(gRandom.Uniform(2., 6.) )

   npeaks = 0
   while (xmax > (xmin + 6. * sigma)) :
      npeaks += 1

      xmin += 3. * sigma  # "mean"
      f.SetParameters(xmin, sigma)

      area = 1. * int( gRandom.Uniform(1., 11.) )
      h.Add(f, area, "")  
      # Or it could be with the "I" option.
      # h.Add(f, area, "I")
      # So, the function would be integrated over 
      # the histogram's bin width, and the area
      # under the curve of the function is added to
      # the histogram.
      # Useful when dealing with probability density 
      # functions or peaks.

      area_sigma_2pi = area / sigma / float( TMath.Sqrt( TMath.TwoPi() ) )

      print(
               "Created : " , f"{xmin:+07.3f}"           , "  |  " ,
            "Area/σ/√2π : " , f"{area_sigma_2pi:+07.3f}" , "  |  " ,
                  "Area : " , f"{area:+07.3f}"           , # "  |  " ,
            )

      # Re-calculating xmin.
      xmin += 3. * sigma
      
   print("\n")
   print(
         f"The total number of created peaks = " + f"{ npeaks }" + 
                                " with sigma = " + f"{ sigma }"     
                              # " with sigma = " + f"{ sigma :+02.3f }" ) 
         )
   print("\n")

   return h
   

# void
def FitAwmi( ) :
   
   h = FitAwmi_Create_Spectrum()
   
   cFit = gROOT.GetListOfCanvases().FindObject("cFit")  # ( TCanvas * )

   if (not cFit) :
      cFit = TCanvas("cFit", "cFit", 10, 10, 1000, 700)
   else:
      cFit.Clear()

   h.Draw("L")


   i, nfound, Bin = [ int() ] * 3 

   nBins = h.GetNbinsX()
   
   source  = array( "d", [ float() ] * nBins )
   dest    = array( "d", [ float() ] * nBins )
   
   #   for (i = 0; i < nBins; i++) {
   for i in range(0, nBins, 1):
      source[i] = h.GetBinContent(i + 1)

   # Note: Default maxpositions = 100.
   s = TSpectrum()  

   # searching for candidate peaks positions
   nfound = s.SearchHighRes(source, dest, nBins, 2., 2., False, 10000, False, 0)

   # filling in the initial estimates of the input parameters
   FixPos = array( "b", [ Bool_t() ]*nfound )
   FixAmp = array( "b", [ Bool_t() ]*nfound )

   #   for (i = 0; i < nfound; i++) {
   for i in range(0, nfound, 1):
      FixAmp[i] = False
      FixPos[i] = False
   
   Amp = array( "d", [ float() ] * nfound )
   
   Pos = s.GetPositionX()  # From 0 ... to (nBins - 1).

   #   for (i = 0; i < nfound; i++) {
   for i in range(0, nfound, 1):
      Bin = 1 + int( Pos[i] + 0.5 )  # The "nearest" bin.
      Amp[i] = h.GetBinContent(Bin)
      



   pfit = TSpectrumFit(nfound)
   pfit.SetFitParameters( 0                         ,
                          (nBins - 1)               ,
                          1000                      ,
                          0.1                       ,
                          pfit.kFitOptimChiCounts   ,
                          pfit.kFitAlphaHalving     ,
                          pfit.kFitPower2           ,
                          pfit.kFitTaylorOrderFirst ,
                          )
   pfit.SetPeakParameters( 2.     ,
                           False  ,
                           Pos    ,
                           FixPos ,
                           Amp    ,
                           FixAmp ,
                           )
   # pfit.SetBackgroundParameters( source[0] ,
   #                               False     ,
   #                               0.        ,
   #                               False     ,
   #                               0.        ,
   #                               False     ,
   #                               )

   pfit.FitAwmi(source)

   Positions         = pfit.GetPositions()
   PositionsErrors   = pfit.GetPositionsErrors()

   Amplitudes        = pfit.GetAmplitudes()
   AmplitudesErrors  = pfit.GetAmplitudesErrors()

   Areas             = pfit.GetAreas()
   AreasErrors       = pfit.GetAreasErrors()



   gROOT.Remove( gROOT.FindObject( "d" ) ) # Prevent »memory leak«.

   d = TH1F( h ) 
   d.SetNameTitle( "d", "" )
   d.Reset( "M" )

   #   for (i = 0; i < nBins; i++) {
   for i in range(0, nBins, 1):
      d.SetBinContent(i + 1, source[i])

   x1 = d.GetBinCenter(1)
   dx = d.GetBinWidth(1)

   sigma     = c_double()
   sigmaErr  = c_double()
   #
   pfit.GetSigma( sigma, sigmaErr )
   #
   sigma     = sigma.value
   sigmaErr  = sigmaErr.value
   
   # Current "TSpectrumFit" needs a "sqrt(2)" correction factor for "sigma".
   sigma     /= TMath.Sqrt2()
   sigmaErr  /= TMath.Sqrt2()
   # Convert "Bin numbers" into "x-axis values".
   sigma     *= dx
   sigmaErr  *= dx
   
   print("\n")
   print( "The total number of found peaks = " ,
          nfound                               ,
          " with sigma = "                     ,
          sigma                                ,
          " (+-"                               ,
          sigmaErr                             ,
          ")"                                  ,
          )
   print("Fit chi^2 = " , pfit.GetChi() )
   print("\n")



   #   for (i = 0; i < nfound; i++) {
   for i in range(0, nfound, 1):

      Bin = 1 + int( Positions[i] + 0.5 )  # The »nearest« Bin.
      Pos[i] = d.GetBinCenter(  Bin )
      Amp[i] = d.GetBinContent( Bin )
      
      # Convert "Bin numbers" into "x-axis values".
      Positions[i] = x1 + Positions[i] * dx
      PositionsErrors[i] *= dx
      Areas[i] *= dx
      AreasErrors[i] *= dx
      
      print( "Found at "
            f" Pos: {Positions[i]:+010.6f}  (+- {PositionsErrors[i]:.6f}  )  | "
            f" Amp: {Amplitudes[i]:+06.2f}  (+- {AmplitudesErrors[i]:.2f} )  | "
            f" Area: {Areas[i]:+010.6f}     (+- {AreasErrors[i]:.6f}      )  | "
             )
      
   d.SetLineColor(kRed)
   d.SetLineWidth(1)
   d.Draw("SAME L")



   pm = h.GetListOfFunctions().FindObject("TPolyMarker") # (TPolyMarker*)
   if pm:
      h.GetListOfFunctions().Remove( pm )
      gROOT.Remove( pm )
   pm = TPolyMarker( nfound, Pos, Amp )

   h.GetListOfFunctions().Add( pm )

   pm.SetMarkerStyle( 23   )
   pm.SetMarkerColor( kRed )
   pm.SetMarkerSize(  1    )



   # Clean up.
   gROOT.Remove( pfit ) 
   gROOT.Remove( s ) 
   del pfit
   del Amp 
   del FixAmp 
   del FixPos 
   del s
   del dest 
   del source 


   return
   


if __name__ == "__main__":
   FitAwmi()
