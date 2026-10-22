## \file
## \ingroup tutorial_math
## \notebook -nodraw
##
## Script illustrating the I/O with Lorentz Vectors of floats, a.k.a:
## TLorentzVector( float, float, float, float)
## The dictionary for LorentzVector's of float is not in libMathCore but in libMathMore,
## therefore it is generated when we parsed its source-file with CLING in .C-version. 
## In python, we just need be sure that ROOT has libMathMore.so loaded. If not, you'll
## have to get ROOT version>"6.30". 
## 
##
## In your python interpreter, to run this script you must do:
##
## ~~~{.cpp}
## IP[0] %run mathcoreVectorFloatIO.py
## IP[1] runIt()
## ~~~
##
## \macro_code
##
## \author Lorenzo Moneta
## \translator P. P.


import ROOT

TRandom = ROOT.TRandom 
TStopwatch = ROOT.TStopwatch 
TSystem = ROOT.TSystem 
TFile = ROOT.TFile 
TTree = ROOT.TTree 
TH1D = ROOT.TH1D 
TCanvas = ROOT.TCanvas 
iostream = ROOT.iostream 
TLorentzVector = ROOT.TLorentzVector 

#math
Math = ROOT.Math
#Vector4D = Math.Vector4D 
XYZTVectorF = Math.XYZTVectorF

#types
Double_t = ROOT.Double_t
Int_t = ROOT.Int_t

#constants
kWhite = ROOT.kWhite

#globals
gPad = ROOT.gPad


# For .C version.
# Now, the dictionary contains the vector's with float types
# So, no need to force dictionary generation.
# You need to run ACLIC with old ROOT version.
# and uncomment these lines below:
# #ifdef __MAKECINT__
# #pragma link C++ class ROOT::Math::PxPyPzE4D<float>+;
# #pragma link C++ class ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<float> >+;
# #pragma link C++ typedef ROOT::Math::XYZTVectorF;
# #endif


# void
def write(n : Int_t) :
   
   print("\n")
   print(30*">")
   print(" write() function")
   print(30*">")
   global R, timer
   R = TRandom()
   timer = TStopwatch()
   
   global f1
   f1 = TFile("mathcoreVectorIO_F.root","RECREATE")
   
   # create tree
   global t1
   t1 = TTree("t1","Tree with new Float LorentzVector")
   
   global v1
   v1 = XYZTVectorF()
   t1.Branch("LV branch","ROOT::Math::XYZTVectorF", v1)
   
   timer.Start()
   #for (int i = 0; i < n; ++i) {
   for i in range(0, n, 1):
      Px = R.Gaus(0,10)
      Py = R.Gaus(0,10)
      Pz = R.Gaus(0,10)
      E = R.Gaus(100,10)
      v1.SetCoordinates(Px,Py,Pz,E)
      t1.Fill()
      
   
   f1.Write()
   timer.Stop()
   print(f" Time for new Float Vector. RealTime = {timer.RealTime() : 5f}, CpuTime = {timer.CpuTime() : 5f} ")
   t1.Print()
   

# void
def read() :
   print("\n") 
   print(30*">")
   print("read() function")
   print(30*">")

   global R, timer
   R = TRandom()
   timer = TStopwatch()
   
   global f1
   f1 = TFile("mathcoreVectorIO_F.root")
   
   # create tree
   global t1
   t1 = f1.Get("t1") # TTree
   
   v1 = XYZTVectorF() 
   t1.SetBranchAddress("LV branch", v1)
   
   timer.Start()
   n = Int_t ( t1.GetEntries() )
   print(f" Tree Entries " , n )
   etot = Double_t(0)
   #for (int i = 0; i < n; ++i) {
   for i in range(0, n, 1):
    
      t1.GetEntry(i)
      etot += v1.E()
      
   
   timer.Stop()
   print(f" Time for new Float Vector. RealTime = {timer.RealTime() : 5f}, CpuTime = {timer.CpuTime() : 5f} ")
   print(f" E average  {n}, etot = {etot:5f}, etot/n =  {etot/Double_t(n) : 5f} ")
   

# void
def runIt() :
   
   #if defined(__CINT__) && !defined(__MAKECINT__)
   if False:
      ROOT.gSystem.Load("libMathCore")
      ROOT.gSystem.Load("libPhysics")
      
      print(f"This tutorial can run only using ACliC, you must run it by doing: ")
      print(f"\t  ROOT.gInterpreter.ProcessLine(\"\"\".L tutorials/math/mathcoreVectorFloatIO.C+\"\"\" ) ")
      print(f"\t  ROOT.gInterpreter.ProcessLine(\"\"\"runIt()\"\"\") ")
   #endif


   nEvents = 100000
   write(nEvents)
   read()
   

# void
def mathcoreVectorFloatIO() :
   #if defined(__CINT__) && !defined(__MAKECINT__)
   if False:
      ROOT.gSystem.Load("libMathCore")
      ROOT.gSystem.Load("libPhysics")
      
      print(f"This tutorial can run only using ACliC, you must run it by doing: ")
      print(f"\t  ROOT.gInterpreter.ProcessLine(\"\"\".L tutorials/math/mathcoreVectorFloatIO.C+\"\"\" ) ")
      print(f"\t  ROOT.gInterpreter.ProcessLine(\"\"\"runIt()\"\"\") ")

   #endif
   


if __name__ == "__main__":
   mathcoreVectorFloatIO()
   #runIt()
