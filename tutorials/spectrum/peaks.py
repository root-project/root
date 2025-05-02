## \file
## \ingroup tutorial_spectrum
## \notebook
##
## Illustrates how to find peaks in histograms.
##
## This script generates a random number of gaussian peaks
## on top of a linear background. Then, it finds them.
## The positions of the peaks are found via TSpectrum-class, and then,
## injected into the parameters of a 'fitter' as 
## its initial values to make a global fit.
## Moreover, the background is computed and drawn on top of 
## the original histogram.
##
## This script, also, can fit "peaks' heights" or "peaks' areas".
## For that you'll have to comment-out or uncomment
## the lines that define `__PEAKS_C_FIT_AREAS__`.
##
## To execute this example, do (in pyroot):
##
## ~~~{.py}
##  IPython[1]: %run peaks.py  # generates 10 peaks by default
##  IPython[2]: peaks(30)      # generates 30 peaks
## ~~~
##
## To execute only the first part of the script(it means without fitting),
## specify a negative value for the number of peaks; e.g.
##
## ~~~{.py}
##  IPython[2]: peaks(-20)     # generates 20 peaks without fitting 
## ~~~
##
## \macro_output
## \macro_image
## \macro_code
##
## \author Rene Brun
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
                   TF1,
                   TH1,
                   TMath,
                   TRandom,
                   TSpectrum,
                   TVirtualFitter,
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




# # # FLAGS # # #
# Comment out the line below, if you want "peaks' heights".
# Uncomment     the line below, if you want "peaks' areas".
#
__PEAKS_C_FIT_AREAS__ = 1 # fit peaks' areas 
#__PEAKS_C_FIT_AREAS__ = 0 # fit peaks' heihts 


npeaks = 30


# Double_t
def fpeaks(x : Double_t, par : Double_t) :

   global result
   result = par[0] + par[1] * x[0]

   #   for (Int_t p = 0; p < npeaks; p++) {
   for p in range(0, npeaks, 1):

      norm   = par[ 3 * p + 2 ] # "height" or "area"
      mean   = par[ 3 * p + 3 ]
      sigma  = par[ 3 * p + 4 ]

      if __PEAKS_C_FIT_AREAS__ :
         norm /= sigma * ( TMath.Sqrt( TMath.TwoPi() ) ) # "area"

      result += norm * TMath.Gaus( x[0], mean, sigma )
      
   return result
   

# void
def peaks(np : Int_t = 10) :

   global npeaks
   npeaks = TMath.Abs( np )

   global h
   h = TH1F( "h", "test", 500, 0, 1000 )
   
   # Generate n peaks at random.
   par = [ Double_t() for _ in range(3000) ]
   par[0] = 0.8
   par[1] = -0.6 / 1000
   
   #   for (p = 0; p < npeaks; p++) {
   for p in range(0, npeaks, 1):

      par[3 * p + 2] = 1                            # "height"
      par[3 * p + 3] = 10    + gRandom.Rndm() * 980 # "mean"
      par[3 * p + 4] = 3 + 2 * gRandom.Rndm()       # "sigma"
    
      if (__PEAKS_C_FIT_AREAS__) :
         # "area"
         par[3 * p + 2] *= par[3 * p + 4] * TMath.Sqrt( TMath.TwoPi() ) 

      

   global f
   f = TF1("f", fpeaks, 0, 1000, 2 + 3 * npeaks)
   f.SetNpx(1000)

   global c_par
   c_par = to_c( par )  
   f.SetParameters(c_par)
   par = to_py( c_par )  

   global c1
   c1 = TCanvas("c1", "c1", 10, 10, 1000, 900)
   c1.Divide(1, 2)
   c1.cd(1)

   h.FillRandom("f", 200000)
   h.Draw()


   # Saving histogram state into h2. 
   global h2
   h2 = h.Clone("h2") # (TH1F*)
   

   # Use TSpectrum to find the peak candidates
   global s
   s = TSpectrum(2 * npeaks)
   global nfound
   nfound = s.Search( h, 2, "", 0.10 )
   #
   print( "Found %d candidate peaks to fit\n" % nfound )
   

   # Estimate background using "TSpectrum.Background" . 
   global hb
   hb = s.Background(h, 20, "same")
   if (hb) :
      c1.Update()

   if (np < 0) : 
      # No Fitting.
      return
   

   c1.cd(2)
   # Estimate linear background using a fitting method.
   #
   global fline
   fline = TF1("fline", "pol1", 0, 1000)
   #
   h.Fit("fline", "qn")
   #
   # Loop on all found peaks. Eliminate peaks at the background level.
   par[0] = fline.GetParameter(0)
   par[1] = fline.GetParameter(1)
   #
   npeaks = 0
   # xpeaks = ctypes.pointer( c_double() ) 
   global xpeaks
   xpeaks = s.GetPositionX() # c_double *
   #   for (p = 0; p < nfound; p++) {
   for p in range(0, nfound, 1):
      xp   = xpeaks[p]
      Bin  = h.GetXaxis().FindBin( xp )
      yp   = h.GetBinContent( Bin )
      if (yp - TMath.Sqrt(yp) < fline.Eval(xp)) :
         continue

      par[ 3 * npeaks + 2 ] = yp # "height"
      par[ 3 * npeaks + 3 ] = xp # "mean"
      par[ 3 * npeaks + 4 ] = 3  # "sigma"

      if (__PEAKS_C_FIT_AREAS__) :
        # "area"
        par[3 * npeaks + 2] *= par[3 * npeaks + 4] * TMath.Sqrt( TMath.TwoPi() )

      npeaks += 1
   # # # 
   print( "Found %d useful peaks to fit\n" % npeaks )


   print( "\n\n >>> Now fitting...: Please, be patient. Could take time.\n" )
   #
   global fit
   fit = TF1("fit", fpeaks, 0, 1000, 2 + 3 * npeaks)
   #
   # We may have more than the default 25 parameters, adding 10 shoud do it.
   TVirtualFitter.Fitter( h2, 10 + 3 * npeaks )
   #
   c_par = to_c( par )  
   fit.SetParameters( c_par)
   par = to_py( c_par )  
   #
   fit.SetNpx(1000)
   #
   h2.Fit("fit")
   


if __name__ == "__main__":
   peaks()
   # peaks(-20)
