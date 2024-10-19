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


#standard library
std = ROOT.std
make_shared = std.make_shared
unique_ptr = std.unique_ptr

#classes
TString = ROOT.TString
TFile = ROOT.TFile
TSpectrum = ROOT.TSpectrum
TPolyMarker = ROOT.TPolyMarker

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
kOrange = ROOT.kOrange
kMagenta = ROOT.kMagenta
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
def SearchHR1() :

   #
   fPositionX = [ Double_t() for _ in range(100) ]
   fPositionY = [ Double_t() for _ in range(100) ]
   fPositionX = array( "d", fPositionX )
   fPositionY = array( "d", fPositionY )
   #
   fNPeaks = 0; # Int_t

   #
   i = Int_t()
   nbins = 1024 # Int_t
   xmin = 0; # Double_t
   xmax = Double_t( nbins ); # Double_t
   a = Double_t()

   #
   source = [ Double_t() for _ in range( nbins ) ]
   dest   = [ Double_t() for _ in range( nbins ) ]
   source = array( "d", source )
   dest = array( "d", dest )
   #
   gROOT.ForceStyle()
   
   Dir = gROOT.GetTutorialDir(); # TString
   file = Dir + TString( "/spectrum/TSpectrum.root" ); # TString
   global f
   f = TFile(file.Data()); # TFile

   global h
   h = f.Get("back2"); # (TH1F *)
   h.SetTitle("High resolution peak searching, number of iterations = 3")
   h.GetXaxis().SetRange(1, nbins)

   global d
   d = TH1F("d", "", nbins, xmin, xmax); # TH1F
   h.Draw("L")
   
   #for (i = 0; i < nbins; i++) {
   for i in range(0, nbins, 1):
      source[i] = h.GetBinContent(i + 1)
   # #   
   h.Draw("L")
   

   #
   global s
   s = TSpectrum(); # TSpectrum
   
   global nfound
   nfound = s.SearchHighRes(source, dest, nbins, 8, 2, True, 3, True, 3)
   global xpeaks
   xpeaks = s.GetPositionX() # Double_t *
   #
   #for (i = 0; i < nfound; i++) {
   for i in range(0, nfound, 1):
      a             = xpeaks[i]
      Bin           = 1 + Int_t(a + 0.5)
      #
      fPositionX[i] = h.GetBinCenter(Bin)
      fPositionY[i] = h.GetBinContent(Bin)
      
   
   global pm
   pm = h.GetListOfFunctions().FindObject("TPolyMarker"); # (TPolyMarker *)
   if (pm)  :
      h.GetListOfFunctions().Remove(pm)
      del pm
   # #   
   pm = TPolyMarker(nfound, fPositionX, fPositionY); # new
   h.GetListOfFunctions().Add(pm)
   #
   pm.SetMarkerStyle(23)
   pm.SetMarkerColor(kRed)
   pm.SetMarkerSize(1.3)

   
   #for (i = 0; i < nbins; i++) {
   for i in range(0, nbins, 1):
      d.SetBinContent(i + 1, dest[i])
   # #   
   d.SetLineColor(kRed)
   d.Draw("SAME")
   

   #
   printf("Found %d candidate peaks\n", nfound)
   #for (i = 0; i < nfound; i++) {
   for i in range(0, nfound, 1):
      printf("posx= %f, posy= %f\n", fPositionX[i], fPositionY[i])
      
   


if __name__ == "__main__":
   SearchHR1()
