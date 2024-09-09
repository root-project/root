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


#standard library
std = ROOT.std
make_shared = std.make_shared
unique_ptr = std.unique_ptr

#classes
TSpectrum = ROOT.TSpectrum
TFile = ROOT.TFile
TMarker = ROOT.TMarker
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
def to_py( c_ls ):
   return list( c_ls )
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
def Background_smooth() :

   i = Int_t()
   nbins = 4096 # Int_t
   xmin = 0; # Double_t
   xmax = nbins; # Double_t

   global source
   source = [ Double_t() for _ in range(nbins) ]
   source = array( "d", source ) 

   gROOT.ForceStyle()
   
   global d1, d2
   d1 = TH1F("d1", "", nbins, xmin, xmax); # TH1F
   d2 = TH1F("d2", "", nbins, xmin, xmax); # TH1F
   
   Dir = gROOT.GetTutorialDir(); # TString
   file = Dir + TString( "/spectrum/TSpectrum.root" ); # TString
   global f
   f = TFile(file.Data()); # TFile

   global back
   back = f.Get("back1"); # (TH1F *)
   back.SetTitle("Estimation of background with noise")
   back.SetAxisRange(3460, 3830)
   back.Draw("L")
   


   # # #
   global s
   s = TSpectrum(); # TSpectrum


   # - - d1 - - # 
   # 
   #for (i = 0; i < nbins; i++) {
   for i in range(0, nbins, 1):
      source[i] = back.GetBinContent(i + 1)
   # # 
   s.Background(
                source,
                nbins,
                6,
                TSpectrum.kBackDecreasingWindow,
                TSpectrum.kBackOrder2,
                False,
                TSpectrum.kBackSmoothing3,
                False,
   )
   #
   #
   #for (i = 0; i < nbins; i++) {
   for i in range(0, nbins, 1):
      d1.SetBinContent(i + 1, source[i])
   # #
   d1.SetLineColor(kRed)
   d1.Draw("SAME L")

   

   # - - d2 - - # 
   #
   #for (i = 0; i < nbins; i++) {
   for i in range(0, nbins, 1):
      source[i] = back.GetBinContent(i + 1)
   # #
   s.Background(
                source,
                nbins,
                6,
                TSpectrum.kBackDecreasingWindow,
                TSpectrum.kBackOrder2,
                True,
                TSpectrum.kBackSmoothing3,
                False,
   )
   #
   #
   #for (i = 0; i < nbins; i++) {
   for i in range(0, nbins, 1):
      d2.SetBinContent(i + 1, source[i])
   # #
   d2.SetLineColor(kBlue)
   d2.Draw("SAME L")
   


if __name__ == "__main__":
   Background_smooth()
