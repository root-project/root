## \file
## \ingroup tutorial_graphs
## \notebook -js
##
## Strips a chart example.
##
## \macro_image
## \macro_code
##
## \author Rene Brun
## \translator P. P.


import ROOT
import ctypes
TCanvas = ROOT.TCanvas 
TDatime = ROOT.TDatime 
TH1F = ROOT.TH1F 
TRandom = ROOT.TRandom 
TStopwatch = ROOT.TStopwatch 
TStyle = ROOT.TStyle 
TSystem = ROOT.TSystem 
##cstdio = ROOT.cstdio 

#classes
TCanvas = ROOT.TCanvas
TH1F = ROOT.TH1F
TGraph = ROOT.TGraph
TLatex = ROOT.TLatex

TCanvas = ROOT.TCanvas
TPaveLabel = ROOT.TPaveLabel
TPavesText = ROOT.TPavesText
TPaveText = ROOT.TPaveText
TText = ROOT.TText
TArrow = ROOT.TArrow
TWbox = ROOT.TWbox
TPad = ROOT.TPad
TBox = ROOT.TBox
TPad = ROOT.TPad
#maths
sin = ROOT.sin
cos = ROOT.cos
sqrt = ROOT.sqrt
exp = ROOT.exp

#types
Double_t = ROOT.Double_t
Float_t = ROOT.Float_t
Int_t = ROOT.Int_t
nullptr = ROOT.nullptr
c_double = ctypes.c_double

#utils
def to_c( ls ):
   return (c_double * len(ls) )( * ls )
def printf(string, *args):
   print( string % args )
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
def seism() :
   
   global sw
   sw = TStopwatch()
   sw.Start()

   # set time offset
   global dtime
   dtime = TDatime()
   gStyle.SetTimeOffset(dtime.Convert())
   

   global c1
   c1 = TCanvas("c1", "Time on axis", 10, 10, 1000, 500)
   c1.SetGrid()

   # One bin = 1 per second. Change it to set-up the time scale.
   bintime = 1
   global ht
   ht = TH1F("ht", "The ROOT seism", 10, 0, 10 * bintime)

   signalval = 1000
   ht.SetMaximum(signalval)
   ht.SetMinimum(-signalval)

   ht.SetStats(False)
   ht.SetLineColor(2)
   ht.GetXaxis().SetTimeDisplay(1)
   ht.GetYaxis().SetNdivisions(520)

   ht.Draw()
   
   #   for (Int_t i = 1; i < 2300; i++) {
   for i in range(1, 2300, 1):

      #======= Building a signal : a noisy damped sine ======
      noise = gRandom.Gaus(0, 120)

      if (i > 700) :
         noise += signalval * sin((i - 700.) * 6.28 / 30) * exp((700. - i) / 300.)

      ht.SetBinContent(i, noise)

      c1.Modified()
      c1.Update()

      # Canvas can be edited during the loop.
      ROOT.gSystem.ProcessEvents()
      


   printf("Real Time = %8.3fs, Cpu Time = %8.3fs\n", sw.RealTime(), sw.CpuTime())
   


if __name__ == "__main__":
   seism()
