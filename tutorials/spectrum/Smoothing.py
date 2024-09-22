## \file
## \ingroup tutorial_spectrum
## \notebook
##
## Example to illustrate smoothing using Markov algorithm (class TSpectrum).
##
## \macro_image
## \macro_code
##
## \author Miroslav Morhac
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
def Smoothing() :

   #
   i = Int_t()
   #
   nbins = 1024 # Int_t
   #
   xmin = 0. ; # Double_t
   xmax = Double_t( nbins ); # Double_t
   #
   source = [ Double_t() for _ in range(nbins) ]


   gROOT.ForceStyle()

   
   Dir = gROOT.GetTutorialDir(); # TString
   file = Dir + TString( "/spectrum/TSpectrum.root" ) ; # TString
   global f
   f = TFile(file.Data()); # TFile


   global h
   h = f.Get("back1"); # (TH1F *)
   h.SetTitle("Smoothed spectrum for m=3")

   
   # Putting bins onto source.
   #
   #for (i = 0; i < nbins; i++) {
   for i in range(0, nbins, 1):
      source[i] = h.GetBinContent(i + 1)
   # #
   h.SetAxisRange(1, 1024)
   h.Draw("L")
   


   # # #
   global s
   s = TSpectrum(); # TSpectrum

   
   #
   global c_source
   c_source = to_c( source )

   # # # 
   s.SmoothMarkov(
                   c_source,  # source     : Double_t*
                   1024,      # ssize      : Int_t
                   3,         # averWindow : Int_t  # Tested on 3 or 7 or 10
   
   ) # -> const char*


   #
   global smooth
   smooth = TH1F("smooth", "smooth", nbins, 0., nbins); # TH1F
   smooth.SetLineColor(kRed)
   #
   #for (i = 0; i < nbins; i++) {
   for i in range(0, nbins, 1):
      smooth.SetBinContent(i + 1, c_source[i])
   # #
   smooth.Draw("L SAME")
   


if __name__ == "__main__":
   Smoothing()
