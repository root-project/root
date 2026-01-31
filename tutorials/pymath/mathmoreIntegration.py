## \file
## \ingroup tutorial_math
## \notebook -nodraw
##
##
## Example on how to use the adaptive integration algorithm in 1Dimension of MathMore.
## It calculates numerically the cumulative integral of a distribution like BreitWigner(in this particular cae).
## To execute this macro, you should type:
##
## ~~~{.py}
## IPython[0] %run mathmoreIntegration.py
## ~~~
## Note:
##      In .C version, you need to compile it with AClic; i.e. use '+'.
##      In .py version, you just need to have installed ROOT version > 6.30.
##
## This tutorial requires to have libMathMore built with ROOT.
## Alternatively, you can use a pre-compiled ROOT version > 6.30.
##
## To build mathmore you need to have a version of GSL >= 1.8 installed in your system.
## The ROOT-configuration will automatically find GSL if the script gsl-config (from GSL) is in your PATH;
## otherwise you need to configure root with the options --gsl-incdir and --gsl-libdir.
## 
## Enjoy it very much as we did writing this tutorial.
##
## \macro_image
## \macro_output
## \macro_code
##
## \authors M. Slawinska, L. Moneta
## \translator P. P.


import ROOT

TMath = ROOT.TMath 
TH1 = ROOT.TH1 
TH1D = ROOT.TH1D
TCanvas = ROOT.TCanvas 
TLegend = ROOT.TLegend 
iostream = ROOT.iostream 
TStopwatch = ROOT.TStopwatch 
TF1 = ROOT.TF1 
#limits = ROOT.limits  #Not implemented in ROOT as namespace.
#TLabel = ROOT.TLabel 

#Math
Math = ROOT.Math
Functor = Math.Functor 
WrappedFunction = Math.WrappedFunction 
#IFunction = Math.IFunction  #Not implemented in ROOT as namespace.
Integrator = Math.Integrator 

#types
Int_t = ROOT.Int_t
Double_t = ROOT.Double_t
nullptr = ROOT.nullptr

#constants
kBlack = ROOT.kBlack
kBlue = ROOT.kBlue
kRed = ROOT.kRed

#standardlibrary
std = ROOT.std


#seamlessly integration
ProcessLine = ROOT.gInterpreter.ProcessLine

#!calculates exact integral of Breit Wigner distribution
#!and compares with existing methods

# double
def exactIntegral( a : Double_t, b : Double_t) :
   
   return (TMath.ATan(2*b)- TMath.ATan(2*a))/ TMath.Pi()
   


global nc
nc = 0 
# double
def func( x : Double_t):
   globals()["nc"] +=1
   return TMath.BreitWigner(x)
   

#Note:
# TF1 requires the function to have the ( )( double *, double *) signature
# We will use an empty_slot to mimic that signature.
# double
def func2(x : Double_t, empty_slot = nullptr ):
   globals()["nc"] +=1
   return TMath.BreitWigner(x[0])
   




# void
def testIntegPerf(x1: Double_t, x2 : Double_t, n: Int_t  = 100000):
   global nc # Number of calls
   print(f"\n\n***************************************************************\n")
   print(f"Test integration performances in interval [ " , x1 , " , " , x2 , " ]\n\n")
   
   timer = TStopwatch()
   
   dx = (x2-x1)/Double_t(n)
   
   global f1 
   #f1 = ROOT.Math.Functor1D["ROOT::Math::IGenFunction"]( TMath.BreitWigner)
   f1 = ROOT.Math.WrappedFunction[""] (func)
   
   timer.Start()
   global ig
   ig = ROOT.Math.Integrator ( f1 )
   s1 = 0.0
   nc = 0 
   #for (int i = 0; i < n; ++i) {
   for i in range(1, n, 1):
      x = x1 + dx*i
      s1 += ig.Integral(x1,x)
      
   timer.Stop()
   print(f"Time using ROOT::Math::Integrator        :\t" , round(timer.RealTime(), 5))
   print(f"number of function calls nc/n = " , int(nc/n))
   pr = std.cout.precision(18);
   print("Integral Value for : \n   func(x)  = ", round(s1,5), f"from {x1} to {x2}")
   print("                    [ a.k.a BreitWigner-Distribution ]")
   std.cout.precision(pr)
   
   
   #Note:
   #      This is faster but cannot measure the number of function calls.
   global fBW
   #fBW = TF1("fBW","TMath::BreitWigner(x)",x1, x2);  
   fBW = TF1("fBW",func2,x1, x2,0)
   
   timer.Start()
   nc = 0
   s2 = 0
   #for (int i = 0; i < n; ++i) {
   for i in range(0, n, 1):
      x = x1 + dx*i
      s2 += fBW.Integral(x1, x)
      
   timer.Stop()
   print(f"Time using TF1::Integral :\t\t\t" , round(timer.RealTime(), 5))
   print(f"Number of function calls nc/n = " , int(nc/n))
   #Note:
   #      In .C version, here-below print(s1) instead of print(s2). 
   #      In .py version, we corret this type-error.
   pr = std.cout.precision(18) 
   print("Integral Value for: \n   func2(x, empty_slot=nullptr) = ", round(s2,5), f"from {x1} to {x2}")
   print("                    [ a.k.a BreitWigner-Distribution ]")
   std.cout.precision(pr)
   
   
   

# void
def  DrawCumulative(x1: Double_t, x2 : Double_t, n: Int_t  = 100):
   
   print(f"\n\n***************************************************************\n")
   print(f"Drawing cumulatives of BreitWigner in interval [ " , x1 , " , " , x2 , " ]\n\n")
   
   
   dx = (x2-x1)/Double_t(n)

   #exact cumulative
   cum0 = TH1D("cum0", "", n, x1, x2); 
   #for (int i = 1; i <= n; ++i) {
   for i in range(1, n+1, 1):
      x = x1 + dx*i
      cum0.SetBinContent(i, exactIntegral(x1, x))
      
      
   
   # alternative method using ROOT.Math.Functor class
   global f1
   f1 = ROOT.Math.Functor1D( func)
   
   
   global ig
   ig = ROOT.Math.Integrator(f1, ROOT.Math.IntegrationOneDim.kADAPTIVE,1.E-12,1.E-12)
   
   
   global cum1
   cum1 = TH1D("cum1", "", n, x1, x2)
   #for (int i = 1; i <= n; ++i) {
   for i in range(1, n+1, 1): 
      x = x1 + dx*i
      cum1.SetBinContent(i, ig.Integral(x1,x))
      
   
   
   global fBW
   fBW = TF1("fBW","TMath::BreitWigner(x, 0, 1)",x1, x2)
   
   
   global cum2
   cum2 = TH1D("cum2", "", n, x1, x2)
   #for (int i = 1; i <= n; ++i) {
   for i in range(1, n+1, 1):
      x = x1 + dx*i
      cum2.SetBinContent(i, fBW.Integral(x1,x))
      
   
   global cum10, cum20
   cum10 = TH1D("cum10", "", n, x1, x2); #difference between 1 and exact
   cum20 = TH1D("cum23", "", n, x1, x2); #difference between 2 and excact
   #for (int i = 1; i <= n; ++i) {
   for i in range(1, n+1, 1): 
      delta = cum1.GetBinContent(i) - cum0.GetBinContent(i)
      delta2 = cum2.GetBinContent(i) - cum0.GetBinContent(i)
      #print( " diff for " , x , " is " , delta , "  " , cum1.GetBinContent(i) )
      cum10.SetBinContent(i, delta )
      cum10.SetBinError(i, std.numeric_limits["double"].epsilon() * cum1.GetBinContent(i) )
      cum20.SetBinContent(i, delta2 )
      
   
   
   global c1
   c1 = TCanvas("c1","Integration example",20,10,800,500)
   c1.Divide(2,1)
   c1.Draw()
   
   cum0.SetLineColor(kBlack)
   cum0.SetTitle("BreitWigner - the cumulative")
   cum0.SetStats(False)
   cum1.SetLineStyle(2)
   cum2.SetLineStyle(3)
   cum1.SetLineColor(kBlue)
   cum2.SetLineColor(kRed)
   c1.cd(1)
   cum0.DrawCopy("h")
   cum1.DrawCopy("same")
   #cum2.DrawCopy("same")
   cum2.DrawCopy("same")
   
   c1.cd(2)
   cum10.SetTitle("Difference")
   cum10.SetStats(False)
   cum10.SetLineColor(kBlue)
   cum10.Draw("e0")
   cum20.SetLineColor(kRed)
   cum20.Draw("hsame")
   
   global l
   l = TLegend(0.11, 0.8, 0.7 ,0.89)
   l.AddEntry(cum10, "GSL integration - analytical ")
   l.AddEntry(cum20, "TF1::Integral  - analytical ")
   l.Draw()
   
   
   c1.Update()
   print(f"\n***************************************************************\n")
   
   
   


#void
def mathmoreIntegration(a : Double_t = -2., b : Double_t = 2.):

   DrawCumulative(a, b)
   testIntegPerf(a, b)
   



if __name__ == "__main__":
   mathmoreIntegration()
