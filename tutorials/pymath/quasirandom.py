## \file
## \ingroup tutorial_math
## \notebook -js
##
## Example of how to generaete quasi-random numbers with
## the methods: pseudo-random, sobol and niederreiter.  
## Last, we compare them with each other.
## >>>   Time for gRandom 
## >>>   Real time 0:00:00, CP time 0.040
## >>>   Time for Sobol 
## >>>   Real time 0:00:00, CP time 0.040
## >>>   Time for Niederreiter 
## >>>   Real time 0:00:00, CP time 0.030
## >>>   number of empty bins for pseudo-random =  31139
## >>>   number of empty bins for  sobol 	=  30512
## >>>   number of empty bins for  niederreiter-base-2 	=  30512
## Niederreiter has a better performance.
## Enjoy!
##
## \macro_image
## \macro_output
## \macro_code
##
## \author Lorenzo Moneta
## \translator P. P.


import ROOT
import ctypes

TH2 = ROOT.TH2 
TH2D = ROOT.TH2D 
TCanvas = ROOT.TCanvas 
TStopwatch = ROOT.TStopwatch 
iostream = ROOT.iostream 

#math
Math = ROOT.Math
QuasiRandom = Math.QuasiRandom 
Random = Math.Random 
RandomMT = Math.RandomMT 
QuasiRandomSobol = Math.QuasiRandomSobol
QuasiRandomNiederreiter = Math.QuasiRandomNiederreiter


#types
Double_t = ROOT.Double_t
Int_t = ROOT.Int_t
c_double = ctypes.c_double

#globals
gROOT = ROOT.gROOT
gPad = ROOT.gPad

#utils
def to_c(ls):
   return (c_double * len(ls))(*ls)



# int
def quasirandom(n : Int_t = 10000, skip : Int_t = 0) :
   
   global h0, h1, h2
   h0 = TH2D("h0","Pseudo-random Sequence",200,0,1,200,0,1)
   h1 = TH2D("h1","Sobol Sequence",200,0,1,200,0,1)
   h2 = TH2D("h2","Niederrer Sequence",200,0,1,200,0,1)
   
   r0 = RandomMT()
   # quasi random numbers need to be created giving the dimension of the sequence
   # in this case we generate 2-d sequence
   
   r1 = QuasiRandomSobol(2)
   r2 = QuasiRandomNiederreiter(2)
   
   # generate n random points
   
   x = [ Double_t() ] * 2
   x = to_c(x)
   w = TStopwatch()
   w.Start()
   #for (int i = 0; i < n; ++i)  {
   for i in range(0, n, 1):
      r0.RndmArray(2,x)
      h0.Fill(x[0],x[1])
      
   print(f"Time for gRandom ")
   w.Print()
   
   w.Start()
   if( skip>0 ): r1.Skip(skip)
   #for (int i = 0; i < n; ++i)  {
   for i in range(0, n, 1):
      r1.Next(x)
      h1.Fill(x[0],x[1])
      
   print(f"Time for Sobol ")
   w.Print()
   
   w.Start()
   if( skip>0 ): r2.Skip(skip)
   #for (int i = 0; i < n; ++i)  {
   for i in range(0, n, 1):
      r2.Next(x)
      h2.Fill(x[0],x[1])
      
   print(f"Time for Niederreiter ")
   w.Print()
   
   global c1
   c1 = TCanvas("c1","Random sequence",600,1200)
   c1.Divide(1,3)
   c1.cd(1)
   h0.Draw("COLZ")
   c1.cd(2)
   
   # check uniformity
   h1.Draw("COLZ")
   c1.cd(3)
   h2.Draw("COLZ")
   gPad.Update()
   
   # test number of empty bins
   
   nzerobins0 = 0
   nzerobins1 = 0
   nzerobins2 = 0
   #for (int i = 1; i <= h1->GetNbinsX(); ++i) {
   #  for (int j = 1; j <= h1->GetNbinsY(); ++j) {
   for i in range(1, h1.GetNbinsX() + 1, 1):
      for j in range(1, h1.GetNbinsY() + 1, 1):
         if (h0.GetBinContent(i,j) == 0 ): nzerobins0 += 1
         if (h1.GetBinContent(i,j) == 0 ): nzerobins1 += 1
         if (h2.GetBinContent(i,j) == 0 ): nzerobins2 += 1
         
      
   
   print(f"number of empty bins for pseudo-random = " , nzerobins0)
   print(f"number of empty bins for " , r1.Name() , "\t= " , nzerobins1)
   print(f"number of empty bins for " , r2.Name() , "\t= " , nzerobins2)
   
   iret = 0
   if (nzerobins1 >= nzerobins0 ): iret += 1
   if (nzerobins2 >= nzerobins0 ): iret += 2
   return iret
   
   


if __name__ == "__main__":
   quasirandom()
   # To avoid potential memeory leak in case you run this script again.
   gROOT.Remove(h0)
   gROOT.Remove(h1)
   gROOT.Remove(h2)
 
   #However, it crashes after three consecutive runs.
