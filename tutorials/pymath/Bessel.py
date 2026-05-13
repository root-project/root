## \file
## \ingroup tutorial_math
## \notebook
##
##
## Show different kinds of the Bessel functions available in ROOT
## To execute this macro type:
##
## ~~~{.py}
## IP[1] %run Bessel.py
## ~~~
##
## It will create one canvas with a representation
## of the  cylindrical and spherical Bessel functions;
## the so called regular and modified Bessel functions.
## Check-out:
##            https://en.wikipedia.org/wiki/Bessel_function
##            https://mathworld.wolfram.com/BesselFunction.html

## \macro_image
## \macro_code
##
## \author Magdalena Slawinska
## \translator P. P.


import ROOT

TCanvas = ROOT.TCanvas 
TF1 = ROOT.TF1 
TMath = ROOT.TMath 
TLegend = ROOT.TLegend 
TLegendEntry = ROOT.TLegendEntry 
##Riostream = ROOT.Riostream  #Not implemented in cppyy.
TAxis = ROOT.TAxis 
TPaveLabel = ROOT.TPaveLabel 
TSystem = ROOT.TSystem 
##cmath = ROOT.cmath   #Not implemented in cppyy.

#math
Math = ROOT.Math 
##IFunction = Math.IFunction #Not implemented in Math.

#constants
kBlack = ROOT.kBlack

#globals
gPad = ROOT.gPad



# void
def Bessel() :
   # R__LOAD_LIBRARY(libMathCore.so);
   ROOT.gSystem.Load("libMathCore.so")
   
   global DistCanvas
   DistCanvas = TCanvas("DistCanvas", "Bessel functions example", 10, 10, 800, 600)
   DistCanvas.SetFillColor(17)
   DistCanvas.Divide(2, 2)
   DistCanvas.cd(1)
   gPad.SetGrid()
   gPad.SetFrameFillColor(19)
   global leg
   leg = TLegend(0.75, 0.7, 0.89, 0.89)
   
   n = 5; # number of functions in each pad
   # drawing the set of Bessel J functions
   #BP: TF1 JBessel[5]
   #Not to use: 
   # JBessel = [ TF1() ] * 5
   # because it creates a list with the same object 
   #Instead:
   global JBessel
   JBessel = [ TF1() for i in range(5)]
   #for (int nu = 0; nu < n; nu++) {
   for nu in range(0, n, 1) :
      JBessel[nu] = TF1("J_0", "ROOT::Math::cyl_bessel_j([0],x)", 0, 10)
      JBessel[nu].SetParameters(nu, 0.0)
      JBessel[nu].SetTitle(""); #"Bessel J functions")
      JBessel[nu].SetLineStyle(1)
      JBessel[nu].SetLineWidth(3)
      JBessel[nu].SetLineColor(nu + 1)
      
   JBessel[0].GetXaxis().SetTitle("x")
   JBessel[0].GetXaxis().SetTitleSize(0.06)
   JBessel[0].GetXaxis().SetTitleOffset(.7)
   
   # setting the title in a label style
   global p1
   p1 = TPaveLabel(.0, .90, (.0 + .50), (.90 + .10),
                   "Bessel J functions", "NDC")
   p1.SetFillColor(0)
   p1.SetTextFont(22)
   p1.SetTextColor(kBlack)
   
   # setting the legend
   leg.AddEntry(JBessel[0].DrawCopy(), " J_0(x)", "l")
   leg.AddEntry(JBessel[1].DrawCopy("same"), " J_1(x)", "l")
   leg.AddEntry(JBessel[2].DrawCopy("same"), " J_2(x)", "l")
   leg.AddEntry(JBessel[3].DrawCopy("same"), " J_3(x)", "l")
   leg.AddEntry(JBessel[4].DrawCopy("same"), " J_4(x)", "l")
   
   leg.Draw()
   p1.Draw()
   
   #------------------------------------------------
   DistCanvas.cd(2)
   gPad.SetGrid()
   gPad.SetFrameFillColor(19)
   
   global leg2
   leg2 = TLegend(0.75, 0.7, 0.89, 0.89)
   #------------------------------------------------
   # Drawing Bessel k
   global KBessel 
   KBessel = [ TF1() for i in range(5)]
   #for (int nu = 0; nu < n; nu++) {
   for nu in range(0, n, 1):
      KBessel[nu] = TF1("J_0", "ROOT::Math::cyl_bessel_k([0],x)", 0, 10)
      KBessel[nu].SetParameters(nu, 0.0)
      #KBessel[nu].SetTitle("Bessel K functions")
      KBessel[nu].SetTitle("") # doesn't interfere with TPaveLabel title. See below p2.
      KBessel[nu].SetLineStyle(1)
      KBessel[nu].SetLineWidth(3)
      KBessel[nu].SetLineColor(nu + 1)
      
   KBessel[0].GetXaxis().SetTitle("x")
   KBessel[0].GetXaxis().SetTitleSize(0.06)
   KBessel[0].GetXaxis().SetTitleOffset(.7)
   
   # setting title
   global p2
   p2 = TPaveLabel(.0, .90, (.0 + .50), (.90 + .10),
   "Bessel K functions", "NDC")
   p2.SetFillColor(0)
   p2.SetTextFont(22)
   p2.SetTextColor(kBlack)
   
   # setting legend
   leg2.AddEntry(KBessel[0].DrawCopy(), " K_0(x)", "l")
   leg2.AddEntry(KBessel[1].DrawCopy("same"), " K_1(x)", "l")
   leg2.AddEntry(KBessel[2].DrawCopy("same"), " K_2(x)", "l")
   leg2.AddEntry(KBessel[3].DrawCopy("same"), " K_3(x)", "l")
   leg2.AddEntry(KBessel[4].DrawCopy("same"), " K_4(x)", "l")
   leg2.Draw()
   p2.Draw()
   #------------------------------------------------
   DistCanvas.cd(3)
   gPad.SetGrid()
   gPad.SetFrameFillColor(19)
   global leg3
   leg3 = TLegend(0.75, 0.7, 0.89, 0.89)
   #------------------------------------------------
   # Drawing Bessel i
   global iBessel 
   iBessel = [ TF1() for i in range(5)]
   #for (int nu = 0; nu <= 4; nu++) {
   for nu in range(0, 4+1, 1):
      iBessel[nu] = TF1("J_0", "ROOT::Math::cyl_bessel_i([0],x)", 0, 10)
      iBessel[nu].SetParameters(nu, 0.0)
      #iBessel[nu].SetTitle("Bessel I functions")
      iBessel[nu].SetTitle("") # Doesn't interfere with TPaveLabel. See below p3.
      iBessel[nu].SetLineStyle(1)
      iBessel[nu].SetLineWidth(3)
      iBessel[nu].SetLineColor(nu + 1)
      
   
   iBessel[0].GetXaxis().SetTitle("x")
   iBessel[0].GetXaxis().SetTitleSize(0.06)
   iBessel[0].GetXaxis().SetTitleOffset(.7)
   
   # setting title
   global p3
   p3 = TPaveLabel(.0, .90, (.0 + .50), (.90 + .10),
   "Bessel I functions", "NDC")
   p3.SetFillColor(0)
   p3.SetTextFont(22)
   p3.SetTextColor(kBlack)
   
   # setting legend
   leg3.AddEntry(iBessel[0].DrawCopy(), " I_0", "l")
   leg3.AddEntry(iBessel[1].DrawCopy("same"), " I_1(x)", "l")
   leg3.AddEntry(iBessel[2].DrawCopy("same"), " I_2(x)", "l")
   leg3.AddEntry(iBessel[3].DrawCopy("same"), " I_3(x)", "l")
   leg3.AddEntry(iBessel[4].DrawCopy("same"), " I_4(x)", "l")
   leg3.Draw()
   p3.Draw()
   #------------------------------------------------
   DistCanvas.cd(4)
   gPad.SetGrid()
   gPad.SetFrameFillColor(19)
   global leg4
   leg4 = TLegend(0.75, 0.7, 0.89, 0.89)
   #------------------------------------------------
   # Drawing sph_bessel
   global jBessel
   jBessel = [ TF1() for i in range(5)]
   #for (int nu = 0; nu <= 4; nu++) {
   for nu in range(0, 4+1, 1): 
      jBessel[nu] = TF1("J_0", "ROOT::Math::sph_bessel([0],x)", 0, 10)
      jBessel[nu].SetParameters(nu, 0.0)
      #jBessel[nu].SetTitle("Bessel j functions")
      jBessel[nu].SetTitle("") # Doesn't interfere with TPaveLabel title. See below p4.
      jBessel[nu].SetLineStyle(1)
      jBessel[nu].SetLineWidth(3)
      jBessel[nu].SetLineColor(nu + 1)
      
   jBessel[0].GetXaxis().SetTitle("x")
   jBessel[0].GetXaxis().SetTitleSize(0.06)
   jBessel[0].GetXaxis().SetTitleOffset(.7)
   
   # setting title
   global p4
   p4 = TPaveLabel(.0, .90, (.0 + .50), (.90 + .10),
   "Bessel j functions", "NDC")
   p4.SetFillColor(0)
   p4.SetTextFont(22)
   p4.SetTextColor(kBlack)
   
   # setting legend
   
   leg4.AddEntry(jBessel[0].DrawCopy(), " j_0(x)", "l")
   leg4.AddEntry(jBessel[1].DrawCopy("same"), " j_1(x)", "l")
   leg4.AddEntry(jBessel[2].DrawCopy("same"), " j_2(x)", "l")
   leg4.AddEntry(jBessel[3].DrawCopy("same"), " j_3(x)", "l")
   leg4.AddEntry(jBessel[4].DrawCopy("same"), " j_4(x)", "l")
   
   leg4.Draw()
   p4.Draw()
   
   DistCanvas.cd()
   


if __name__ == "__main__":
   Bessel()
