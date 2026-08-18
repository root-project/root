# \file
# \ingroup tutorial_math
# \notebook
#
# Tutorial illustrating the 
# multivariate-gaussian-random-number generation.
# 
#
# \macro_image
# \macro_code
#
# \author Jorge Lopez
# \translator P. P.


import ROOT
import ctypes

TH1F = ROOT.TH1F
TH2F = ROOT.TH2F
TCanvas = ROOT.TCanvas

#types
c_double = ctypes.c_double

#utils
def to_c(ls):
   return (c_double * len(ls))( * ls ) 

#void
def multivarGaus():

   #Proper initialization of the GSL Random Engine. 
   global rnd
   rnd = ROOT.Math.GSLRandomEngine() #Create the engine.
   rnd.Initialize() # Initializes the engine.
   
   dim = 3
   pars = [ 0, 0, 0.5 ] # [dim]
   genpars = [ 0, 0, 0 ] # [dim]
   cov = [ 1.0, -0.2, 0.0, -0.2, 1.0, 0.5, 0.0, 0.5, 0.75 ] #[dim * dim]
   
   c_pars = to_c( pars )
   c_genpars = to_c( genpars )
   c_cov = to_c( cov )

   global hX, hY, hZ
   hX = TH1F("hX", "hX;x;Counts", 100, -5, 5)
   hY = TH1F("hY", "hY;y;Counts", 100, -5, 5)
   hZ = TH1F("hZ", "hZ;z;Counts", 100, -5, 5)
   
   global hXY, hXZ, hYZ
   hXY = TH2F("hXY", "hXY;x;y;Counts", 100, -5, 5, 100, -5, 5)
   hXZ = TH2F("hXZ", "hXZ;x;z;Counts", 100, -5, 5, 100, -5, 5)
   hYZ = TH2F("hYZ", "hYZ;y;z;Counts", 100, -5, 5, 100, -5, 5)

   
   MAX = 10000
   #for (int evnts = 0; evnts < MAX; ++evnts) {
   for evnts in range(0, MAX, 1):

      rnd.GaussianND(dim, c_pars, c_cov, c_genpars)

      x = c_genpars[0]
      y = c_genpars[1]
      z = c_genpars[2]

      hX.Fill(x)
      hY.Fill(y)
      hZ.Fill(z)

      hXY.Fill(x, y)
      hXZ.Fill(x, z)
      hYZ.Fill(y, z)
      
   
   global c
   c =  TCanvas("c", "Multivariate gaussian random numbers")
   c.Divide(3, 2)

   c.cd(1)
   hX.Draw()

   c.cd(2)
   hY.Draw()

   c.cd(3)
   hZ.Draw()

   c.cd(4)
   hXY.Draw("COL")

   c.cd(5)
   hXZ.Draw("COL")

   c.cd(6)
   hYZ.Draw("COL")

   ## In case you run this script again.
   #ROOT.gROOT.DeleteAll()
   #ROOT.gROOT.Remove(c)

if __name__ == "__main__":
   multivarGaus()    
