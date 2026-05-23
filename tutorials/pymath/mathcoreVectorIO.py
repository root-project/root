## \file
## \ingroup tutorial_math
## \notebook -nodraw
##
## Example of how to make I/O(input-output) through a TTree with
## some mathcore Lorentz Vectors, and how to compare with a TLorentzVector.
## A ROOT tree is written and read in both: using either a XYZTVector or a 
## TLorentzVector.
##
## To execute the macro type in your IPython interpreter:
##
## ~~~{.py}
## IP[0] %run  mathcoreVectorIO.py
## ~~~
##
## \macro_output
## \macro_code
##
## \author Lorenzo Moneta
## \translator P. P.


import ROOT

TRandom2 = ROOT.TRandom2 
TStopwatch = ROOT.TStopwatch 
TSystem = ROOT.TSystem 
TFile = ROOT.TFile 
TTree = ROOT.TTree 
TH1D = ROOT.TH1D 
TCanvas = ROOT.TCanvas 
iostream = ROOT.iostream 
TLorentzVector = ROOT.TLorentzVector 

#
TVector3 = ROOT.TVector3

#math module
Math = ROOT.Math
#Vector4D = Math.Vector4D 
XYZTVector = Math.XYZTVector
TVector3 

#types
Double_t = ROOT.Double_t
Int_t = ROOT.Int_t





#types
Int_t = ROOT.Int_t



# void
def write(n : Int_t) :
   
   global R, timer
   R = TRandom2()
   timer = TStopwatch()
   
   R.SetSeed(1)
   timer.Start()
   s = 0
   #for (int i = 0; i < n; ++i) {
   for i in range(0, n, 1):
      s += R.Gaus(0,10)
      s += R.Gaus(0,10)
      s += R.Gaus(0,10)
      s += R.Gaus(100,10)
      
   
   timer.Stop()
   print( s/Double_t(n) )
   print(" Time for Random gen " , timer.RealTime() , "  " , timer.CpuTime() )
   
   
   global f1
   f1 = TFile("mathcoreVectorIO_1.root","RECREATE")
   
   # create tree
   global t1
   t1 = TTree("t1","Tree with new LorentzVector")
   
   global v1
   v1 = XYZTVector()
   t1.Branch("LV branch","ROOT::Math::XYZTVector", v1)
   
   R.SetSeed(1)
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
   print(f" Time for new Vector " , timer.RealTime() , "  " , timer.CpuTime())
   
   t1.Print()
   
   # create tree with old LV
   
   global f2, t2
   f2 = TFile("mathcoreVectorIO_2.root","RECREATE")
   t2 = TTree("t2","Tree with TLorentzVector")
   
   global v2
   v2 = TLorentzVector()
   TLorentzVector.Class().IgnoreTObjectStreamer()
   TVector3.Class().IgnoreTObjectStreamer()
   
   t2.Branch("branch = TLV","TLorentzVector", v2,16000,2)
   
   R.SetSeed(1)
   timer.Start()
   #for (int i = 0; i < n; ++i) {
   for i in range(0, n, 1):
      Px = R.Gaus(0,10)
      Py = R.Gaus(0,10)
      Pz = R.Gaus(0,10)
      E = R.Gaus(100,10)
      v2.SetPxPyPzE(Px,Py,Pz,E)
      t2.Fill()
      
   
   f2.Write()
   timer.Stop()
   print(f" Time for old Vector " , timer.RealTime() , "  " , timer.CpuTime())
   t2.Print()
   


# void
def read() :
   
   global R, timer, f1
   R = TRandom()
   timer = TStopwatch()
   f1 = TFile("mathcoreVectorIO_1.root")
   
   # create tree
   global t1
   t1 = f1.Get("t1") # TTree
   
   v1 = 0
   t1.SetBranchAddress("LV branch", v1)
   
   timer.Start()
   n = Int_t( t1.GetEntries() )
   print(f" Tree Entries " , n)
   etot = Double_t(0) 
   #for (int i = 0; i < n; ++i) {
   for i in range(0, n, 1):
      t1.GetEntry(i)
      etot += v1.Px()
      etot += v1.Py()
      etot += v1.Pz()
      etot += v1.E()
      
   timer.Stop()
   print(f" Time for new Vector " , timer.RealTime() , "  " , timer.CpuTime())
   
   print(f" TOT average : n = " , n , "\t " , etot/double(n))
   
   # create tree with old LV
   global f2, t2
   f2 = TFile("mathcoreVectorIO_2.root")
   t2 = f2.Get("t2") # TTree
   
   v2 = 0
   t2.SetBranchAddress("TLV branch", v2)
   
   timer.Start()
   n = Int_t( t2.GetEntries() )
   print(f" Tree Entries " , n)
   etot = 0
   #for (int i = 0; i < n; ++i) {
   for i in range(0, n, 1):
      t2.GetEntry(i)
      etot += v2.Px()
      etot += v2.Py()
      etot += v2.Pz()
      etot += v2.E()
      
   
   timer.Stop()
   print(f" Time for old Vector " , timer.RealTime() , "  " , timer.CpuTime())
   print(f" TOT average:\t" , etot/Double_t(n))
   

# void
def mathcoreVectorIO() :
   
   nEvents = 100000
   write(nEvents)
   read()
   


if __name__ == "__main__":
   mathcoreVectorIO()
