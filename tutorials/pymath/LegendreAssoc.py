## \file
## \ingroup tutorial_math
## \notebook
##
## Example describing how to use the different kinds of 
## Associate Legendre Polynomials of ROOT::Math implementation.
##
## To execute this script type in your python interpreter: 
##
## ~~~{.py}
## IP[0]: %run LegendreAssoc.py
## ~~~
##
## This draws common graphs for the first 5
## Associate Legendre Polynomials $P_{n}^{m}$.
## And also, some graphs of the 
## Spherical Associate Legendre Polynomials $P_{m}^{n}$.
## At last, their corresponding integrals go through the conventional
## range [-1, 1] and are calculated. 
##
## Enjoy it for your course of Electromagnetism and Quantum Mechanics.
##
## \macro_image
## \macro_output
## \macro_code
##
## \author Magdalena Slawinska
## \translator P. P.


import ROOT

TMath = ROOT.TMath 
TF1 = ROOT.TF1 
TCanvas = ROOT.TCanvas 
#Riostream = ROOT.Riostream 
TLegend = ROOT.TLegend 
TLegendEntry = ROOT.TLegendEntry 
#cmath = ROOT.cmath 
TSystem = ROOT.TSystem 

#math
Math = ROOT.Math
#IFunction = Math.IFunction 

#types
Double_t = ROOT.Double_t

#constants
kWhite = ROOT.kWhite

#globals
gPad = ROOT.gPad



# void
def LegendreAssoc() :

   ROOT.gSystem.Load("libMathMore.so")
   
   print("Drawing associate Legendre Polynomials...")

   global Canvas
   Canvas = TCanvas("DistCanvas", "Associate Legendre polynomials", 10, 10, 800, 500)
   Canvas.Divide(2,1)

   global leg1, leg2
   leg1 = TLegend(0.5, 0.7, 0.8, 0.89)
   leg2 = TLegend(0.5, 0.7, 0.8, 0.89)
   
   #-------------------------------------------
   #drawing the set of Legendre functions
   # First, initializing ROOT::Math::assoc_legendre functions.
   global L
   L = [ TF1() for _ in range(5) ]

   L[0].__assign__( TF1 ( "L_0", "ROOT::Math::assoc_legendre(1, 0,x)", -1, 1) )
   L[1].__assign__( TF1 ( "L_1", "ROOT::Math::assoc_legendre(1, 1,x)", -1, 1) )
   L[2].__assign__( TF1 ( "L_2", "ROOT::Math::assoc_legendre(2, 0,x)", -1, 1) )
   L[3].__assign__( TF1 ( "L_3", "ROOT::Math::assoc_legendre(2, 1,x)", -1, 1) )
   L[4].__assign__( TF1 ( "L_4", "ROOT::Math::assoc_legendre(2, 2,x)", -1, 1) )
   
   # Second, initializing ROOT::Math::sph_legendre functions.
   global SL
   SL = [ TF1() for _ in range(5) ]

   SL[0].__assign__( TF1( "SL_0", "ROOT::Math::sph_legendre(1, 0,x)", -TMath.Pi(), TMath.Pi() ) )
   SL[1].__assign__( TF1( "SL_1", "ROOT::Math::sph_legendre(1, 1,x)", -TMath.Pi(), TMath.Pi() ) )
   SL[2].__assign__( TF1( "SL_2", "ROOT::Math::sph_legendre(2, 0,x)", -TMath.Pi(), TMath.Pi() ) )
   SL[3].__assign__( TF1( "SL_3", "ROOT::Math::sph_legendre(2, 1,x)", -TMath.Pi(), TMath.Pi() ) )
   SL[4].__assign__( TF1( "SL_4", "ROOT::Math::sph_legendre(2, 2,x)", -TMath.Pi(), TMath.Pi() ) )
   
   
   #Third, setting-up our canvas and pad for the Associate Legendre Polynomials.
   Canvas.cd(1)
   gPad.SetGrid()
   gPad.SetFillColor(kWhite)
 
   #Once the set-up has been done for one function, the rest will follow up.
   L[0].SetMaximum(3)
   L[0].SetMinimum(-2)
   L[0].SetTitle("Associate Legendre Polynomials")
   #for (int nu = 0; nu < 5; nu++) {
   for nu in range(0, 5, 1):
       
      L[nu].SetLineStyle(1)
      L[nu].SetLineWidth(2)
      L[nu].SetLineColor(nu+1) 
      
   
   #Fourth, writing our legends.
   leg1.AddEntry(L[0].DrawCopy(), " P^{1}_{0}(x)", "l")
   leg1.AddEntry(L[1].DrawCopy("same"), " P^{1}_{1}(x)", "l")
   leg1.AddEntry(L[2].DrawCopy("same"), " P^{2}_{0}(x)", "l")
   leg1.AddEntry(L[3].DrawCopy("same"), " P^{2}_{1}(x)", "l")
   leg1.AddEntry(L[4].DrawCopy("same"), " P^{2}_{2}(x)", "l")
   leg1.Draw()
   
   #Fifth, we repeat third, fourth, and five step for the Spherical Legendre Polynomials functions.
   Canvas.cd(2)
   gPad.SetGrid()
   gPad.SetFillColor(kWhite)
   SL[0].SetMaximum(1)
   SL[0].SetMinimum(-1)
   SL[0].SetTitle("Spherical Legendre Polynomials")
   
   #for (int nu = 0; nu < 5; nu++) {
   for nu in range(0, 5, 1):
      SL[nu].SetLineStyle(1)
      SL[nu].SetLineWidth(2)
      SL[nu].SetLineColor(nu+1)
      
   
   # 
   leg2.AddEntry(SL[0].DrawCopy(), " P^{1}_{0}(x)", "l")
   leg2.AddEntry(SL[1].DrawCopy("same"), " P^{1}_{1}(x)", "l")
   leg2.AddEntry(SL[2].DrawCopy("same"), " P^{2}_{0}(x)", "l")
   leg2.AddEntry(SL[3].DrawCopy("same"), " P^{2}_{1}(x)", "l")
   leg2.AddEntry(SL[4].DrawCopy("same"), " P^{2}_{2}(x)", "l")
   leg2.Draw()
   
   
   #Sixth, the integration process.
   #Using the method .Integral for each function will do.
   
   print(f"Calculating integrals of the Associate Legendre Polynomials over the range [-1, 1]")
   integral = [ Double_t() for _ in range(5) ]
   #for (int nu = 0; nu < 5; nu++) {
   for nu in range(0, 5, 1):
      integral[nu] = L[nu].Integral(-1.0, 1.0)
      print(f"Integral [-1,1] for Associated Legendre Polynomial of Degree {nu:d} \t = \t {integral[nu]:5f} ")
      
   



if __name__ == "__main__":
   LegendreAssoc()
