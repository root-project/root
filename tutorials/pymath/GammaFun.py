## \file
## \ingroup tutorial_math
## \notebook
##
## Example showing how to use the major special math functions (gamma, beta, erf) within ROOT.
## To execute the macro type in a Python interpreter:
##
## ~~~{.py}
## IP[0]: %run GammaFun.py
## ~~~
##
## It will create one canvas with the representation
## of the tgamma, lgamma, beta, erf and erfc functions.
##
## \macro_image
## \macro_code
##
## \author Magdalena Slawinska
## \translator P. P.


import ROOT

TMath = ROOT.TMath 
TF1 = ROOT.TF1 
TF2 = ROOT.TF2 
TSystem = ROOT.TSystem 
TCanvas = ROOT.TCanvas 
TStyle = ROOT.TStyle 
TPaveLabel = ROOT.TPaveLabel 
TAxis = ROOT.TAxis 
TH1 = ROOT.TH1 
TH1F = ROOT.TH1F

#math
Math = ROOT.Math

#types
Double_t = ROOT.Double_t

#constants
kWhite = ROOT.kWhite
kBlack = ROOT.kBlack
kBlue = ROOT.kBlue
kRed = ROOT.kRed

#globals
gPad = ROOT.gPad
gStyle = ROOT.gStyle
gROOT = ROOT.gROOT



# void
def GammaFun() :
   
   gStyle.SetOptStat(0)
   
   global f1a, f2a, f3a, f4a, fb
   f1a = TF1("Gamma(x)","ROOT::Math::tgamma(x)",-2,5)
   f2a = TF1("f2a","ROOT::Math::lgamma(x)",0,10)
   f3a = TF2("Beta(x)","ROOT::Math::beta(x, y)",0,0.1, 0, 0.1)
   f4a = TF1("erf(x)","ROOT::Math::erf(x)",0,5)
   f4b = TF1("erfc(x)","ROOT::Math::erfc(x)",0,5)
   
   global c1
   c1 = TCanvas("c1", "Gamma and related functions",800,700)
   
   c1.Divide(2,2)
   
   c1.cd(1)
   gPad.SetGrid()
   
   #Setting the title in a label style.
   global p1
   p1 = TPaveLabel(.1,.90, (.1+.50),(.90+.10),"ROOT::Math::tgamma(x)", "NDC")
   p1.SetFillColor(0)
   p1.SetTextFont(22)
   p1.SetTextColor(kBlack)
   
   #Setting the graph.
   # Draw its axis first. Use a TH1-object to draw the frame.
   global h
   h = TH1F("htmp","",500,-2,5)
   h.SetMinimum(-20)
   h.SetMaximum(20)
   h.GetXaxis().SetTitleSize(0.06)
   h.GetXaxis().SetTitleOffset(.7)
   h.GetXaxis().SetTitle("x")
   
   h.Draw()
   
   # Draw the functions 3-times in separate ranges to avoid singularities.
   f1a.SetLineWidth(2)
   f1a.SetLineColor(kBlue)
   
   f1a.SetRange(-2,-1)
   f1a.DrawCopy("same")
   
   f1a.SetRange(-1,0)
   f1a.DrawCopy("same")
   
   f1a.SetRange(0,5)
   f1a.DrawCopy("same")
   
   p1.Draw()
   
   c1.cd(2)
   gPad.SetGrid()
   
   # Repeat the process.
   global p2
   p2 = TPaveLabel(.1,.90, (.1+.50),(.90+.10),"ROOT::Math::lgamma(x)", "NDC")
   p2.SetFillColor(0)
   p2.SetTextFont(22)
   p2.SetTextColor(kBlack)
   f2a.SetLineColor(kBlue)
   f2a.SetLineWidth(2)
   f2a.GetXaxis().SetTitle("x")
   f2a.GetXaxis().SetTitleSize(0.06)
   f2a.GetXaxis().SetTitleOffset(.7)
   f2a.SetTitle("")
   f2a.Draw()
   p2.Draw()
   
   c1.cd(3)
   gPad.SetGrid()
   
   # Repeat the process.
   global p3
   p3 = TPaveLabel(.1,.90, (.1+.50),(.90+.10),"ROOT::Math::beta(x, y)", "NDC")
   p3.SetFillColor(0)
   p3.SetTextFont(22)
   p3.SetTextColor(kBlack)
   f3a.SetLineWidth(2)
   f3a.GetXaxis().SetTitle("x")
   f3a.GetXaxis().SetTitleOffset(1.2)
   f3a.GetXaxis().SetTitleSize(0.06)
   f3a.GetYaxis().SetTitle("y")
   f3a.GetYaxis().SetTitleSize(0.06)
   f3a.GetYaxis().SetTitleOffset(1.5)
   f3a.SetTitle("")
   f3a.Draw("surf1");#option for a 3-dim plot
   p3.Draw()
   
   c1.cd(4)
   gPad.SetGrid()
   
   # Repeat the process
   global p4
   p4 = TPaveLabel(.1,.90, (.1+.50),(.90+.10),"erf(x) and erfc(x)", "NDC")
   p4.SetFillColor(0)
   p4.SetTextFont(22)
   p4.SetTextColor(kBlack)
   f4a.SetTitle("erf(x) and erfc(x)")
   f4a.SetLineWidth(2)
   f4b.SetLineWidth(2)
   f4a.SetLineColor(kBlue)
   f4b.SetLineColor(kRed)
   f4a.GetXaxis().SetTitleSize(.06)
   f4a.GetXaxis().SetTitleOffset(.7)
   f4a.GetXaxis().SetTitle("x")
   f4a.Draw()
   f4b.Draw("same");#option for a multiple graph plot
   f4a.SetTitle("")
   p4.Draw()
   
   
   # To avoid potential memory leak.
   gROOT.Remove( h ) 
if __name__ == "__main__":
   GammaFun()
