## \file
## \ingroup tutorial_math
## \notebook -js
## 
## Demo for quantiles.
##
## \macro_image
## \macro_code
##
## \authors Rene Brun, Eddy Offermann
## \translator P. P.


import ROOT
import ctypes

#classes
TGraph = ROOT.TGraph
TH1F = ROOT.TH1F
TCanvas = ROOT.TCanvas
TLegend = ROOT.TLegend

#types
Double_t = ROOT.Double_t
Float_t = ROOT.Float_t
Int_t = ROOT.Int_t
c_double = ctypes.c_double

   
#colors
kRed = ROOT.kRed
kMagenta = ROOT.kMagenta
kBlue = ROOT.kBlue
kCyan = ROOT.kCyan
kGreen = ROOT.kGreen
kYellow = ROOT.kYellow
kOrange = ROOT.kOrange

#utils
def to_c(ls):
   return (c_double * len(ls) ) ( * ls)

def to_py(c_ls):
   return list(c_ls)

#globlas
gPad = ROOT.gPad
gROOT = ROOT.gROOT

# void
def quantiles() :

   nq = 100
   nshots = 10

   xq = [ Double_t() for _ in range(nq) ]  # position where to compute the quantiles in [0,1]
   yq = [ Double_t() for _ in range(nq) ]  # array to contain the quantiles

   # Loading xq points.
   #for (Int_t i=0;i<nq;i++) xq[i] = Float_t(i+1)/nq
   for i in range(0, nq, 1): xq[i] = Float_t(i+1)/nq

   # Convertion to C-types
   c_xq = to_c(xq)
   c_yq = to_c(yq)

   global gr70, gr90, gr98
   gr70 = TGraph(nshots)
   gr90 = TGraph(nshots)
   gr98 = TGraph(nshots)
   
   global h
   h = TH1F("h","demo quantiles",50,-3,3)
   
   #for (Int_t shot=0;shot<nshots;shot++) {
   for shot in range(0, nshots, 1):
      h.FillRandom("gaus",50)
      h.GetQuantiles(nq,c_yq,c_xq)
      gr70.SetPoint(shot,shot+1,c_yq[70])
      gr90.SetPoint(shot,shot+1,c_yq[90])
      gr98.SetPoint(shot,shot+1,c_yq[98])
      
   
   #show the original histogram in the top pad
   global c1
   c1 = TCanvas("c1","demo quantiles",10,10,600,900)
   c1.Divide(1,3)
   c1.cd(1)
   h.SetFillColor(38)
   h.Draw()
   
   # show the final quantiles in the middle pad
   c1.cd(2)
   gPad.SetGrid()

   global gr
   gr = TGraph(nq,c_xq,c_yq)
   gr.SetTitle("final quantiles")
   gr.SetMarkerStyle(21)
   gr.SetMarkerColor(kRed)
   gr.SetMarkerSize(0.3)
   gr.Draw("ap")
   
   # show the evolution of some  quantiles in the bottom pad
   c1.cd(3)
   gPad.DrawFrame(0,0,nshots+1,3.2)
   gPad.SetGrid()
   gr98.SetMarkerStyle(22)
   gr98.SetMarkerColor(kRed)
   gr98.Draw("lp")
   gr90.SetMarkerStyle(21)
   gr90.SetMarkerColor(kBlue)
   gr90.Draw("lp")
   gr70.SetMarkerStyle(20)
   gr70.SetMarkerColor(kMagenta)
   gr70.Draw("lp")

   # add a legend
   global legend
   legend = TLegend(0.85,0.74,0.95,0.95)
   legend.SetTextFont(72)
   legend.SetTextSize(0.05)
   legend.AddEntry(gr98," q98","lp")
   legend.AddEntry(gr90," q90","lp")
   legend.AddEntry(gr70," q70","lp")
   legend.Draw()
   
   
   #To avoid potential memory leak in case of re-running this script.
   #gROOT.Remove( h ) 
   


if __name__ == "__main__":
   quantiles()
