## \file
## \ingroup tutorial_graphs
## \notebook
##
## An example of TGraphTime-class use showing how the class could be used to visualize
## a set of particles with their time stamp in a MonteCarlo program.
## Enjoy the demo! It is pretty cool actually.
##
## \macro_image
## \macro_code
##
## \author Rene Brun
## \translator P. P.


import ROOT
import ctypes
TArrow = ROOT.TArrow 
TGraphTime = ROOT.TGraphTime 
TMarker = ROOT.TMarker 
TMath = ROOT.TMath 
TPaveLabel = ROOT.TPaveLabel 
TRandom3 = ROOT.TRandom3 

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
def Form(string, *args):
   return string % args

#constants
kYellow = ROOT.kYellow

kYellow = ROOT.kYellow
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
def gtime2(nsteps : Int_t = 200, np : Int_t = 5000) :

   if (np > 5000) :
      np = 5000

   color = [ Int_t() for _ in range(5000) ]
   cosphi, sinphi, speed = [ [ Double_t() for _ in range(5000) ] for _ in range(3) ]    
   r = TRandom3()

   xmin = 0; xmax = 10; ymin = -10; ymax = 10

   g = TGraphTime(nsteps, xmin, ymin, xmax, ymax)
   g.SetTitle("TGraphTime demo 2;X;Y")

   i, s = Int_t(), Int_t()
   fact = xmax / Double_t(nsteps)

   # calculate some object parameters
   #   for (i = 0; i < np; i++) {
   for i in range(0, np, 1): 
      speed[i] = r.Uniform(0.5, 1)
      phi = r.Gaus(0, TMath.Pi() / 6.)
      cosphi[i] = fact * speed[i] * TMath.Cos(phi)
      sinphi[i] = fact * speed[i] * TMath.Sin(phi)
      rc = r.Rndm()
      color[i] = kRed
      if (rc > 0.3) :
         color[i] = kBlue
      if (rc > 0.7) :
         color[i] = kYellow


   global m_list, PaveLabel_list
   m_list= []
   PaveLabel_list= []
   # fill the TGraphTime step by step  
   #   for (s = 0; s < nsteps; s++) {
   #      for (i = 0; i < np; i++) {
   for s in range(0, nsteps, 1):
      for i in range(0, np, 1):
         xx = s * cosphi[i]
         if (xx < xmin) :
            continue
         yy = s * sinphi[i]
         m = TMarker(xx, yy, 25)
         m.SetMarkerColor(color[i])
         m.SetMarkerSize(1.5 - s / (speed[i] * nsteps))
         g.Add(m, s)
         m_list.append( m )
      my_PaveLabel = TPaveLabel(.70, .92, .98, .99, Form("shower at %5.3f nsec", 3. * s / nsteps), "brNDC")    
      g.Add( my_PaveLabel , s)
      PaveLabel_list.append( my_PaveLabel )
      
      
   g.Draw()
   


if __name__ == "__main__":
   gtime2()
