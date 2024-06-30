## \file
## \ingroup tutorial_math
## \notebook -js
## 
##
## Example showing how to write and read a std::vector of ROOT::Math::LorentzVector 's
## through a ROOT tree.
## 
## In the write() function, a variable number of track Vectors is generated
## according to a Poisson distribution. $\Phi and $\eta$, their angular-coordinates, 
## are uniformily distributed over with a random 4-momentum $P$.
##
## On the other hand, 
## in the read() function, the vectors are read backwards 
## and their content is analysed by; and also, some relevant information, 
## such as the number of tracks per event or the 4-momentum track 
## distributions $p_t$, are displayed in a canvas.
##
## To execute the script type:
##
## ~~~{.py}
##   IP[0]: %run  mathcoreVectorCollection.py
## ~~~
## in the IPython interpreter.
## Enjoy!
##
## \macro_image
## \macro_output
## \macro_code
##
## \author Andras Zsenei
## \translator P. P.


import ROOT

TRandom = ROOT.TRandom 
TStopwatch = ROOT.TStopwatch 
TSystem = ROOT.TSystem 
TFile = ROOT.TFile 
TTree = ROOT.TTree 
TH1D = ROOT.TH1D 
TCanvas = ROOT.TCanvas 

#Math
Math = ROOT.Math
TMath = ROOT.TMath 
iostream = ROOT.iostream 
#Vector3D = Math.Vector3D 
#Vector4D = Math.Vector4D 

#standard library
std = ROOT.std

#types
Int_t = ROOT.Int_t
Double_t = ROOT.Double_t





#Note: 
#    In .C version:
#    CLING does not understand some files included by LorentzVector header.


# double
def write(n : Int_t) :
   global R, timer
   R = TRandom()
   timer = TStopwatch()
   
   global f1
   f1 = TFile("mathcoreLV.root","RECREATE")
   
   # create tree
   global t1
   t1 = TTree("t1","Tree with new LorentzVector")
   
   global tracks, pTracks
   tracks = std.vector["ROOT::Math::XYZTVector"]
   #BP: pTracks = std.vector[ROOT::Math::XYZTVector] (tracks)
   #pTracks = std.vector["ROOT::Math::XYZTVector"] (tracks)
   pTracks = tracks
   #t1.Branch("tracks",
   #          """std::vector<
   #                ROOT::Math::LorentzVector<
   #                   ROOT::Math::PxPyPzE4D<
   #                      double
   #                   > 
   #                > 
   #             >
   #          """,
   #          pTracks)
   
   # set pi+ (pion+) mass
   M = 0.13957; 
   
   timer.Start()
   Sum = 0
   #for(int i = 0; i < n; i++) {
   for i in range(0, n, 1):
      nPart = R.Poisson(5)
      pTracks.clear()
      pTracks.reserve(nPart)
      #for(int j = 0; j < nPart; j++) {
      for j in range(0, nPart, 1):
         px = R.Gaus(0,10)
         py = R.Gaus(0,10)
         pt = sqrt(px*px +py*py)
         eta = R.Uniform(-3,3)
         phi = R.Uniform(0.0, 2*TMath.Pi() )
         vcyl = RhoEtaPhiVector( pt, eta, phi)
         # set energy
         E = sqrt( vcyl.R() * vcyl.R() + M*M)
         q = XYZTVector( vcyl.X(), vcyl.Y(), vcyl.Z(), E )
         # fill track vector
         pTracks.push_back(q)
         # evaluate sum of components to check
         Sum += q.x() + q.y() + q.z() + q.t()
         
      t1.Fill()
      
   
   f1.Write()
   timer.Stop()
   print(f" Time for new Vector " , timer.RealTime() , "  " , timer.CpuTime())
   
   t1.Print()
   return Sum
   

# double
def read() :
   global R, timer
   R = TRandom()
   timer = TStopwatch()
   
   global h1, h2, h3, h4, h5, h6
   h1 = TH1D("h1","total event  energy ",100,0,1000.)
   h2 = TH1D("h2","Number of track per event",21,-0.5,20.5)
   h3 = TH1D("h3","Track Energy",100,0,200)
   h4 = TH1D("h4","Track Pt",100,0,100)
   h5 = TH1D("h5","Track Eta",100,-5,5)
   h6 = TH1D("h6","Track Cos(theta)",100,-1,1)
   
   f1 = TFile("mathcoreLV.root")
   
   # create tree
   t1 = f1.Get("t1") # TTree
   
   pTracks = nullptr
   t1.SetBranchAddress("tracks", pTracks)
   
   timer.Start()
   n = t1.GetEntries() # Int_t
   print(f" Tree Entries " , n)
   Sum = Double_t(0)
   #for(int i = 0; i < n; i++) {
   for i in range(0, n, 1):
      t1.GetEntry(i)
      ntrk = pTracks.size()
      h3.Fill(ntrk)
      q = XYZTVector()
      #for(int j = 0; j < ntrk; j++) {
      for j in range(0, ntrk, 1):
         v = pTracks[j]
         q += v
         h3.Fill(v.E())
         h4.Fill(v.Pt())
         h5.Fill(v.Eta())
         h6.Fill(cos(v.Theta()))
         Sum += v.x() + v.y() + v.z() + v.t()
         
      h1.Fill(q.E() )
      h2.Fill(ntrk)
      
   
   timer.Stop()
   print(f" Time for new Vector " , timer.RealTime() , "  " , timer.CpuTime())
   
   global c1
   c1 = TCanvas("c1","demo of Trees",10,10,600,800)
   c1.Divide(2,3)
   
   c1.cd(1)
   h1.Draw()
   c1.cd(2)
   h2.Draw()
   c1.cd(3)
   h3.Draw()
   c1.cd(3)
   h3.Draw()
   c1.cd(4)
   h4.Draw()
   c1.cd(5)
   h5.Draw()
   c1.cd(6)
   h6.Draw()
   
   return Sum
   

# int
def mathcoreVectorCollection() :
   
   nEvents = 10000
   s1 = write(nEvents)
   s2 = read()
   
   if fabs(s1-s2) > s*11.E-15 :
      print(f"ERROR: Found difference in Vector when reading  ( " , s1 , " != " , s2 , " diff = " , fabs(s1-s2) , " ) ")
      return -1
      
   return 0
   

# int
def main() :
   return mathcoreVectorCollection()
   



if __name__ == "__main__":
   mathcoreVectorCollection()
