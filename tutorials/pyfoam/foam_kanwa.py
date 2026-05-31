# \file
# \ingroup FOAM's_python_tutorials
# \notebook -js
# This program can be execute from the command line as folows:
#
# ~~~{.py}
#      ipython3 foam_kanwa.py
# ~~~
#
# \macro_code
#
# \author Stascek Jadach
# \translator P. P.

import ROOT
import ctypes

#Riostream = 		 ROOT.Riostream
TFoam = 		 ROOT.TFoam
TCanvas = 		 ROOT.TCanvas
TH2 = 		 ROOT.TH2
TMath = 		 ROOT.TMath
TFoamIntegrand = 		 ROOT.TFoamIntegrand
TRandom3 = 		 ROOT.TRandom3
TH2D = ROOT.TH2D

Double_t = ROOT.Double_t
c_double = ctypes.c_double

exp = ROOT.exp

#_____________________________________________________________________________
def sqr(x):
   return x*x
   
#_____________________________________________________________________________
def Camel2(nDim, Xarg):
   if type(nDim) != int and type(Xarg) != list : 
      raise TypeError("nDim or Xarg aren't good enugh")
   # 2-dimensional distribution for Foam, normalized to one (within 1e-5)
   x=Xarg[0]
   y=Xarg[1]
   GamSq= sqr(0.100e0)
   Dist= 0.
   Dist += exp( -( sqr(x-1./3) + sqr(y-1./3) )/GamSq ) / GamSq / TMath.Pi()
   Dist += exp( -( sqr(x-2./3) + sqr(y-2./3) )/GamSq ) / GamSq / TMath.Pi()
   return 0.5*Dist
   
#_____________________________________________________________________________

def foam_kanwa():
   print("--- kanwa started ---")
   hst_xy =  TH2D("hst_xy" , "x-y plot", 50,0,1.0, 50,0,1.0)
   MCvect = [Double_t() ]*2 # 2-dim vector generated in the MC run
   c_MCvect = (c_double*2)(*MCvect) 
   #
   PseRan =  TRandom3()     # Create random number generator
   PseRan.SetSeed(4357)
   FoamX =  TFoam("FoamX")  # Create Simulator
   FoamX.SetkDim(2)         # No. of dimensions, obligatorynot
   FoamX.SetnCells(500)     # Optionally No. of cells, default=2000
   FoamX.SetRhoInt(Camel2)  # Set 2-dim distribution, included below
   FoamX.SetPseRan(PseRan)  # Set random number generator
   FoamX.Initialize()       # Initialize simulator, may take time...
   #
   # visualising generated distribution
   cKanwa =  TCanvas("cKanwa","Canvas for plotting",600,600)
   cKanwa.cd()
   # From now on FoamX is ready to generate events
   nshow=5000
   #for loop in range( 100000 ): # Bad for memory
   loop = 0
   while( loop < 100000 ):
      loop += 1
      FoamX.MakeEvent()            # generate MC event
      FoamX.GetMCvect(c_MCvect)     # get generated vector (x,y)
      x = c_MCvect[0]
      y = c_MCvect[1]

      if(loop<10):
         print(f"loop: {loop} ", "(x,y) = ( ",  x , ", ",  y , " )")
      if(loop%10000 == 0):
         print(f"loop: {loop} ", "(x,y) = ( ",  x , ", ",  y , " )")
      hst_xy.Fill(x,y)
   
      # live plot
      if(loop == nshow):
         nshow += 5000
         hst_xy.Draw("lego2")
         cKanwa.Update()
         
   # end of loop
   #
   hst_xy.Draw("lego2")  # final plot
   cKanwa.Update()
   cKanwa.Draw()
   #
   MCresult = c_double(0.0) 
   MCerror = c_double(0.0) 
   global gFoamX
   gFoamX = FoamX 
   FoamX.GetIntegMC( MCresult, MCerror)  # get MC integral, should be one
   print(" MCresult=", MCresult, " +- " ,  MCerror )
   print("--- kanwa ended ---")
   
   return 0
   
foam_kanwa()
