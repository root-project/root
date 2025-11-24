## \file
## \ingroup tutorial_graphics
## \notebook
##
## This tutorial illustrates the special contour options:
##
##   - "AITOFF"     : Draws a contour via an AITOFF projection
##   - "MERCATOR"   : Draws a contour via an Mercator projection
##   - "SINUSOIDAL" : Draws a contour via an Sinusoidal projection
##   - "PARABOLIC"  : Draws a contour via an Parabolic projection
##   - "MOLLWEIDE"  : Draws a contour via an Mollweide projection
##
## \macro_image
## \macro_code
##
## \author Olivier Couet (from an original macro sent by Ernst-Jan Buis)
## \translator P. P.


import ROOT
import ctypes

#classes
TCanvas = ROOT.TCanvas
TH2F = ROOT.TH2F
TPaveLabel = ROOT.TPaveLabel
TPavesText = ROOT.TPavesText
TPaveText = ROOT.TPaveText
TText = ROOT.TText
TArrow = ROOT.TArrow
TWbox = ROOT.TWbox
TPad = ROOT.TPad
TBox = ROOT.TBox
TPad = ROOT.TPad

#standard library
std = ROOT.std
ifstream = std.ifstream

#types
Double_t = ROOT.Double_t
Float_t = ROOT.Float_t
Int_t = ROOT.Int_t
nullptr = ROOT.nullptr
c_double = ctypes.c_double
c_int = ctypes.c_int

#utils
def to_c( ls ):
   return (c_double * len(ls) )( * ls )
def printf(string, *args):
   print( string % args )
def sprintf(buffer, string, *args):
   buffer = string % args 
   return buffer

#constants

#globals
gStyle = ROOT.gStyle
gPad = ROOT.gPad
gRandom = ROOT.gRandom
gBenchmark = ROOT.gBenchmark
gROOT = ROOT.gROOT



# TCanvas
def earth() :
   
   gStyle.SetOptTitle(1)
   gStyle.SetOptStat(0)
   
   global c1
   c1 = TCanvas("c1","earth_projections",700,1000)
   c1.Divide(2,3)
   
   global ha, hm, hs, hp, hw
   ha = TH2F("ha","Aitoff", 180, -180, 180, 179, -89.5, 89.5)
   hm = TH2F("hm","Mercator", 180, -180, 180, 161, -80.5, 80.5)
   hs = TH2F("hs","Sinusoidal",180, -180, 180, 181, -90.5, 90.5)
   hp = TH2F("hp","Parabolic", 180, -180, 180, 181, -90.5, 90.5)
   hw = TH2F("hw","Mollweide", 180, -180, 180, 181, -90.5, 90.5)
   
   global dat
   dat = ROOT.gROOT.GetTutorialDir()
   dat.Append("/graphics/earth.dat")
   dat.ReplaceAll("/./","/")
   
   In = ifstream() 
   In.open(dat.Data())
   # Data have int-type format. Check earth.dat file.
   x = c_int()
   y = c_int()
   while (1) :
      # We read their values.
      In >> x >> y
      if ( not In.good()) : break
      # We load their values.
      ha.Fill( x.value,  y.value,  1)
      hm.Fill( x.value,  y.value,  1)
      hs.Fill( x.value,  y.value,  1)
      hp.Fill( x.value,  y.value,  1)
      hw.Fill( x.value,  y.value,  1)
      
   In.close()
   
   c1.cd(1)
   ha.Draw("aitoff")
   c1.cd(2)
   hm.Draw("mercator")
   c1.cd(3)
   hs.Draw("sinusoidal")
   c1.cd(4)
   hp.Draw("parabolic")
   c1.cd(5)
   hp.Draw("mollweide")
   
   return c1
   


if __name__ == "__main__":
   earth()
