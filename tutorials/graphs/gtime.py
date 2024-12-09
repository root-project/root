## \file
## \ingroup tutorial_graphs
## \notebook
##
## An example of TGraphTime-class.
##
## \macro_image
## \macro_code
##
## \author Rene Brun
## \translator P. P.


import ROOT
import ctypes
TArrow = ROOT.TArrow 
TCanvas = ROOT.TCanvas 
TGraphTime = ROOT.TGraphTime 
TFile = ROOT.TFile
TMath = ROOT.TMath 


TROOT = ROOT.TROOT 
TRandom3 = ROOT.TRandom3 
TText = ROOT.TText 

#classes
TCanvas = ROOT.TCanvas
TMarker = ROOT.TMarker
TMath = ROOT.TMath

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
def gtime(nsteps : Int_t = 500, np : Int_t = 100) :

   if (np > 1000):
      np = 1000

   color = [ Double_t() for _ in range(1000) ]
   rr, phi, dr, size = [ [ Double_t() for _ in range(10000) ] for _ in range(4) ]
   global r
   r = TRandom3() 

   xmin = -10; xmax = 10; ymin = -10; ymax = 10

   global g
   g = TGraphTime(nsteps, xmin, ymin, xmax, ymax)
   g.SetTitle("TGraphTime demo;X domain;Y domain")
   g.SetName("g")

   i, s = Int_t(), Int_t()

   # calculate some object parameters
   #   for (i = 0; i < np; i++) {
   for i in range(0, np, 1):
  
      rr[i] = r.Uniform(0.1 * xmax, 0.2 * xmax)
      phi[i] = r.Uniform(0, 2 * TMath.Pi())
      dr[i] = r.Uniform(0, 1) * 0.9 * xmax / Double_t(nsteps)
      color[i] = kRed

      rc = r.Rndm()
      if (rc > 0.3) :
         color[i] = kBlue
      if (rc > 0.7) :
         color[i] = kYellow

      size[i] = r.Uniform(0.5, 6)

   global m_list, myArrow_list, myPaveLabel_list
   m_list = []
   myArrow_list = []
   myPaveLabel_list = []

   # Fill the TGraphTime step by step.
   #
   #   for (s = 0; s < nsteps; s++) {
   #      for (i = 0; i < np; i++) {
   for s in range(0, nsteps, 1):
      for i in range(0, np, 1):

         newr = rr[i] + dr[i] * s
         newsize = 0.2 + size[i] * TMath.Abs(TMath.Sin(newr + 10))
         newphi = phi[i] + 0.01 * s

         xx = newr * TMath.Cos(newphi)
         yy = newr * TMath.Sin(newphi)

         m = TMarker(xx, yy, 20)
         m.SetMarkerColor(color[i])
         m.SetMarkerSize(newsize)

         m_list.append( m )
         #g.Add(m, s)
         g.Add( m_list[-1], s )

         if (i == np - 1) :
            myArrow = TArrow(xmin, ymax, xx, yy, 0.02, "-|>")
            myArrow_list.append( myArrow )
            #g.Add(myArrow, s)
            g.Add( myArrow_list[-1], s )
         
      myPaveLabel = TPaveLabel(.90, .92, .98, .97, Form("%d", s + 1), "brNDC")
      myPaveLabel_list.append( myPaveLabel )
      #g.Add(myPaveLabel, s)
      g.Add( myPaveLabel_list[-1], s )
      
   #BP:
   g.Draw("")
   
   # save object to a file
   global f
   f = TFile("gtime.root", "recreate")
   #BP:
   g.Write("g")
   f.Close()

   # To view this object in another session you MUST do:
   #   ff = TFile("gtime.root"); #1
   #   ff.Open("gtime.root");    #2
   #   gg = ff.Get("g");         #3 
   #   gg.Draw();                #4
   #   #Afterwards, you should close your file. 
   #   ff.Close();               #5


if __name__ == "__main__":
   gtime()
