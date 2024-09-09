## \file
## \ingroup tutorial_spectrum
## \notebook
##
## Example to illustrate deconvolution function (class TSpectrum).
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
def Deconvolution_wide() :
   #
   i = Int_t()
   #
   nbins = 256 # Int_t
   #
   xmin = 0; # Double_t
   xmax = nbins; # Double_t
   #
   source = [ Double_t() for _ in range(nbins) ]
   response = [ Double_t() for _ in range(nbins) ]

   #
   gROOT.ForceStyle()

   
   Dir = gROOT.GetTutorialDir(); # TString
   file = Dir + TString( "/spectrum/TSpectrum.root" ) ; # TString
   global f
   f = TFile(file.Data()); # TFile


   global h, d
   h = f.Get("decon3"); # (TH1F *)
   d = f.Get("decon_response_wide"); # (TH1F *)
   

   #for (i = 0; i < nbins; i++) {
   for i in range(0, nbins, 1):
      source[i] = h.GetBinContent(i + 1)
      
   #for (i = 0; i < nbins; i++) {
   for i in range(0, nbins, 1):
      response[i] = d.GetBinContent(i + 1)
      
   
   h.SetTitle("Deconvolution of closely positioned "\
              "overlapping peaks using Gold deconvolution "\
              "method")
   h.SetMaximum(50000)
   h.Draw("L")


   # # #
   global s
   s = TSpectrum(); # TSpectrum


   #to_c
   global c_source, c_response
   c_source = to_c( source )
   c_response = to_c( response )

   #
   s.Deconvolution(
                   c_source,
                   c_response,
                   # ssize
                   256,
                   # numberIterations
                   10000,
                   # numberRepetitions
                   1,
                   # boost
                   1,
   )
   
   #for (i = 0; i < nbins; i++) {
   for i in range(0, nbins, 1):
      d.SetBinContent(i + 1, c_source[i])
      
   d.SetLineColor(kRed)
   d.Draw("SAME L")
   


if __name__ == "__main__":
   Deconvolution_wide()
