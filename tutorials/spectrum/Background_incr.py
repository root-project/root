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


#standard library
std = ROOT.std
make_shared = std.make_shared
unique_ptr = std.unique_ptr

#classes
TSpectrum = ROOT.TSpectrum
TFile = ROOT.TFile
TString = ROOT.TString
TCanvas = ROOT.TCanvas
TH1F = ROOT.TH1F
TGraph = ROOT.TGraph
TLatex = ROOT.TLatex

#maths
sin = ROOT.sin
cos = ROOT.cos
sqrt = ROOT.sqrt

#types
Double_t = ROOT.Double_t
Bool_t = ROOT.Bool_t
Float_t = ROOT.Float_t
Int_t = ROOT.Int_t
nullptr = ROOT.nullptr
c_double = ctypes.c_double

#utils
def to_c( ls ):
   return (c_double * len(ls) )( * ls )
def printf(string, *args):
   print( string % args, end="")
def sprintf(buffer, string, *args):
   buffer = string % args 
   return buffer

#constants
kBlue = ROOT.kBlue
kRed = ROOT.kRed
kGreen = ROOT.kGreen

#globals
gStyle = ROOT.gStyle
gPad = ROOT.gPad
gRandom = ROOT.gRandom
gBenchmark = ROOT.gBenchmark
gROOT = ROOT.gROOT



# void
def Background_incr() :

   #
   i = Int_t()
   #
   nbins = 1024 # Int_t
   #
   xmin = 0; # Double_t
   xmax = nbins; # Double_t
   #
   source = [ Double_t() for _ in range(nbins) ]


   gROOT.ForceStyle()

   global d
   d = TH1F("d", "", nbins, xmin, xmax); # TH1F
   
   Dir = gROOT.GetTutorialDir(); # TString
   file = Dir + TString( "/spectrum/TSpectrum.root" ) ; # TString
   global f
   f = TFile(file.Data()); # TFile



   global back
   back = f.Get("back1"); # (TH1F *)
   back.GetXaxis().SetRange(1, nbins)
   back.SetTitle("Estimation of background with increasing window")
   back.Draw("L")
   


   # # # 
   global s
   s = TSpectrum(); # TSpectrum


   
   #for (i = 0; i < nbins; i++) {
   for i in range(0, nbins, 1):
      source[i] = back.GetBinContent(i + 1)
      
   # to_c types
   global c_source
   c_source = to_c( source )
   
   # # #
   # Estimate the background
   #
   # const char *
   s.Background(
                # spectrum # Double_t*
                c_source,
                # ssize # Int_t
                nbins,
                # numberIterations # Int_t
                6,
                # direction # Int_t
                TSpectrum.kBackIncreasingWindow,
                # filterOrder # Int_t
                TSpectrum.kBackOrder2,
                # smoothing # bool
                False,
                # smoothWindow # Int_t
                TSpectrum.kBackSmoothing3,
                # compton # bool 
                False,
   )


   # # # 
   # Draw the estimated background
   #
   #for (i = 0; i < nbins; i++) {
   for i in range(0, nbins, 1):
      d.SetBinContent(i + 1, c_source[i])
   # #   
   d.SetLineColor(kRed)
   d.Draw("SAME L")
   


if __name__ == "__main__":
   Background_incr()
