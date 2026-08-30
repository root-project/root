# \file
# \ingroup tutorial_math
# \notebook
# Test of the TMath::Vavilov distribution
#
# \macro_image
# \macro_code
#
# \author Anna Kreshuk
# \translator P. P.

import ROOT
import ctypes

TMath = ROOT.TMath
TCanvas = ROOT.TCanvas
TRandom = ROOT.TRandom
TGraph = ROOT.TGraph
TF1 = ROOT.TF1
TH1F = ROOT.TH1F

#types
Double_t = ROOT.Double_t
c_double = ctypes.c_double

#utils
def to_c(ls):
   return (c_double * len(ls) )( *ls)

#constants
kBlue = ROOT.kBlue
kRed = ROOT.kRed
kGreen = ROOT.kGreen

#void
def vavilov():

   n = 1000
   xvalues =  [ Double_t() ] * n
   yvalues1 =  [ Double_t() ] * n
   yvalues2 =  [ Double_t() ] * n
   yvalues3 =  [ Double_t() ] * n
   yvalues4 =  [ Double_t() ] * n
   
   global r
   r = TRandom() 
   #for(Int_t i=0 i<n i++)
   for i in range(0, n, 1):
      xvalues[i] = r.Uniform(-2, 10)
      yvalues1[i] = TMath.Vavilov(xvalues[i], 0.3, 0.5)
      yvalues2[i] = TMath.Vavilov(xvalues[i], 0.15, 0.5)
      yvalues3[i] = TMath.Vavilov(xvalues[i], 0.25, 0.5)
      yvalues4[i] = TMath.Vavilov(xvalues[i], 0.05, 0.5)
      
   
   global c1
   c1 =  TCanvas("c1", "Vavilov density")
   c1.SetGrid()
   c1.SetHighLightColor(19)

   # Converting to C-types.
   c_xvalues = to_c(xvalues)
   c_yvalues1 = to_c(yvalues1)
   c_yvalues2 = to_c(yvalues2)
   c_yvalues3 = to_c(yvalues3)
   c_yvalues4 = to_c(yvalues4)

   global gr1, gr2, gr3 ,gr4
   gr1 =  TGraph(n, c_xvalues, c_yvalues1)
   gr2 =  TGraph(n, c_xvalues, c_yvalues2)
   gr3 =  TGraph(n, c_xvalues, c_yvalues3)
   gr4 =  TGraph(n, c_xvalues, c_yvalues4)

   gr1.SetTitle("TMath::Vavilov density")
   gr1.Draw("ap")
   gr2.Draw("psame")
   gr2.SetMarkerColor(kRed)
   gr3.Draw("psame")
   gr3.SetMarkerColor(kBlue)
   gr4.Draw("psame")
   gr4.SetMarkerColor(kGreen)
   
   global f1
   f1 =  TF1("f1", "TMath::Vavilov(x, 0.3, 0.5)", -2, 10)
   
   global hist
   hist =  TH1F("vavilov", "vavilov", 100, -2, 10)
   #for (int i=0 i<10000 i++)
   N_entries = 10000
   for i in range(0, N_entries, 1):
      hist.Fill(f1.GetRandom())
      
   hist.Scale(1/1200.)
   hist.Draw("same")
   
if __name__ == "__main__":
   vavilov() 
