# \file
# \ingroup tutorial_FOAM
# \notebook -nodraw
# Demonstrate the TFoam class.
#
#  To run this macro type from ROOT command line
#
# ~~~{.py}
#  ipython3 [0] %run foam_demo.py
# ~~~
#
# \macro_code
#
# \author Stascek Jadach
# \translator P. P.


import ROOT
import gc #garbage collector
import ctypes

TFile = 		 ROOT.TFile
TFoam = 		 ROOT.TFoam
TH1 = 		 ROOT.TH1
TMath = 		 ROOT.TMath
TFoamIntegrand = 		 ROOT.TFoamIntegrand
TRandom3 = 		 ROOT.TRandom3

TH1D = ROOT.TH1D

long = ROOT.long
Double_t  = ROOT.Double_t  
c_double = ctypes.c_double
c_ptr_double = ctypes.POINTER(c_double)

sqrt = ROOT.Math.sqrt
exp = ROOT.Math.exp

Declare = ROOT.gInterpreter.Declare

Declare('''
class TFDISTR: public TFoamIntegrand {
public:
   TFDISTR(){};
   Double_t Density(int nDim, Double_t *Xarg){
   // Integrand for mFOAM
   Double_t Fun1,Fun2,R1,R2;
   Double_t pos1=1e0/3e0;
   Double_t pos2=2e0/3e0;
   Double_t Gam1= 0.100e0;  // as in JPC
   Double_t Gam2= 0.100e0;  // as in JPC
   Double_t sPi = sqrt(TMath::Pi());
   Double_t xn1=1e0;
   Double_t xn2=1e0;
   int i;
   R1=0;
   R2=0;
   for(i = 0 ; i<nDim ; i++){
      R1=R1+(Xarg[i] -pos1)*(Xarg[i] -pos1);
      R2=R2+(Xarg[i] -pos2)*(Xarg[i] -pos2);
      xn1=xn1*Gam1*sPi;
      xn2=xn2*Gam2*sPi;
   }
   R1   = sqrt(R1);
   R2   = sqrt(R2);
   Fun1 = exp(-(R1*R1)/(Gam1*Gam1))/xn1;  // Gaussian delta-like profile
   Fun2 = exp(-(R2*R2)/(Gam2*Gam2))/xn2;  // Gaussian delta-like profile
   return 0.5e0*(Fun1+ Fun2);
}
  ClassDef(TFDISTR,1) //Class of testing functions for FOAM
};
   ''')
TFDISTR = ROOT.TFDISTR

class TFDISTR_Py( TFoamIntegrand ):
   #public:
   def __init__(self):
      super().__init__()
      pass 
   def Density(self, nDim = int(), Xarg = list() ):
      # Integrand for mFOAM
      Fun1, Fun2, R1, R2 = [Double_t() for i in range(4)]
      pos1=1e0/3e0
      pos2=2e0/3e0
      Gam1= 0.100e0  # as in JPC
      Gam2= 0.100e0  # as in JPC
      sPi = sqrt(TMath.Pi())
      xn1=1e0
      xn2=1e0
      R1=0
      R2=0
      for i in range(nDim):
         R1=R1+(Xarg[i] -pos1)*(Xarg[i] -pos1)
         R2=R2+(Xarg[i] -pos2)*(Xarg[i] -pos2)
         xn1=xn1*Gam1*sPi
         xn2=xn2*Gam2*sPi
      
      R1   = sqrt(R1)
      R2   = sqrt(R2)
      Fun1 = exp(-(R1*R1)/(Gam1*Gam1))/xn1  # Gaussian delta-like profile
      Fun2 = exp(-(R2*R2)/(Gam2*Gam2))/xn2  # Gaussian delta-like profile

      return 0.5e0*(Fun1+ Fun2) 


def foam_demo():

   RootFile = TFile("foam_demo.root","RECREATE","histograms")
   loop = long()
   MCresult , MCerror , MCwt = [c_double(0.0) for i in range(3) ]
   #MCerror = c_double(0.0)
   #-----------------------------------------
   NevTot = 50000 # Total MC statistics
   kDim = 2 # total dimension
   nCells = 500 # Number of Cells
   nSampl = 200 # Number of MC events per cell in build-up
   nBin = 8 # Number of bins in build-up
   OptRej = 1 # Wted events for OptRej=0 wt=1 for OptRej=1 (default)
   OptDrive = 2 # (D=2) Option, type of Drive =0,1,2 for TrueVol,Sigma,WtMax
   EvPerBin = 25 # Maximum events (equiv.) per bin in buid-up
   Chat = 1 # Chat level
   #-----------------------------------------
   PseRan =  TRandom3() # Create random number generator
   FoamX =  TFoam("FoamX") # Create Simulator
   # Whichever works fine: Choose according to your interest performance.
   #                       Either memory performance or agility in programming. 
   #rho =  TFDISTR() # type(rho) is TFoamIntegrand and rho is defined at root
   rho =  TFDISTR_Py() # type(rho) is TFoamIntegrand and rho is defined at python
   PseRan.SetSeed(4357)
   #-----------------------------------------
   print(" Demonstration Program for Foam version ", FoamX.GetVersion(), "    ")
   FoamX.SetkDim(        kDim)      # Mandatory!!!
   FoamX.SetnCells(      nCells)    # optional
   FoamX.SetnSampl(      nSampl)    # optional
   FoamX.SetnBin(        nBin)      # optional
   FoamX.SetOptRej(      OptRej)    # optional
   FoamX.SetOptDrive(    OptDrive)  # optional
   FoamX.SetEvPerBin(    EvPerBin)  # optional
   FoamX.SetChat(        Chat)      # optional
   #-----------------------------------------
   FoamX.SetRho(rho)
   FoamX.SetPseRan(PseRan)
   FoamX.Initialize() # Initialize simulator
   FoamX.Write("FoamX")     # Writing Foam on the disk, TESTING PERSISTENCY!!!
   #-----------------------------------------
   nCalls = FoamX.GetnCalls() 
   print("====== Initialization done, entering MC loop")
   #-----------------------------------------
   #print(" About to start MC loop: ")
   # important: MCvect has to be a *double type to be able to use in FoamX.GetMCvect
   MCvect = [Double_t()]*kDim # vector generated in the MC run
   c_MCvect = (ctypes.c_double * kDim)(*MCvect)
   #-----------------------------------------
   hst_Wt =  TH1D("hst_Wt" , "Main weight of Foam", 25, 0, 1.25)
   hst_Wt.Sumw2()
   #-----------------------------------------
   for loop in range(NevTot) :
      #===============================
      FoamX.MakeEvent()              # generate MC event
      #===============================
      # note: MCvect_c instead of MCvect 
      #       type(MCvect_c) is double*
      #       type(MCvect) is a list of double
      FoamX.GetMCvect( c_MCvect ) 
      MCwt = FoamX.GetMCwt()
      hst_Wt.Fill(MCwt, 1.0)
      # print first 15 loops straight and ...
      if(loop<150):
         print("MCwt = ", MCwt, ",  ")
         print("MCvect = ")
         for k in range( kDim): 
            print("        |", f"{c_MCvect[k]:f}", "|")
         print("")
      # ... print every 100000 loops by.  
      if( ((loop)%100000)==0 ):
         print(" loop= ", loop)
         
      
   
   #-----------------------------------------
   
   print( "====== Events generated. Entering Finale Step ======")
   
   hst_Wt.Print("all")
   eps = 0.0005
   Effic = Double_t()
   WtMax , AveWt , Sigma = [c_double(0.0) for i in range(3)]
   IntNorm , Errel = [c_double(0.0) for i in range(2)]

   FoamX.Finalize(   IntNorm, Errel)     # final printout
   FoamX.GetIntegMC( MCresult, MCerror)  # get MC intnegral
   FoamX.GetWtParams(eps, AveWt, WtMax, Sigma) # get MC wt parameters

   Effic = AveWt.value / WtMax.value
   RelErr = MCerror.value / MCresult.value
   Dispersion = Sigma.value / AveWt.value

   print("================================================================")
   print(f" MCresult= {MCresult.value} +- {MCerror.value}   RelErr= {RelErr}")
   print(f" Dispersion/<wt>= {Dispersion}")
   print(" <wt>/WtMax= ", Effic , ", for epsilon = ", eps)
   print(" nCalls (initialization only) =   " ,  nCalls )
   print("================================================================")
   
   del MCvect
   gc.collect() # garbage collector

   #
   RootFile.ls()
   RootFile.Write()
   RootFile.Close()
   print(" End of Demonstration Program  ")
   
   return 0 # End of demo.

   
if __name__ == "__main__":
   foam_demo()
#foam_demo()
