{
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*
//*-*  This program creates :
//*-*    - a one dimensional histogram
//*-*    - a two dimensional histogram
//*-*    - a profile histogram
//*-*    - a memory-resident ntuple
//*-*
//*-*  These objects are filled with some random numbers and saved on a file.
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

  gROOT->Reset();

// Create a new ROOT binary machine independent file.
// Note that this file may contain any kind of ROOT objects, histograms,
// pictures, graphics objects, detector geometries, tracks, events, etc..
// This file is now becoming the current directory.

#ifdef ClingWorkAroundMissingImplicitAuto
  TNtuple *ntuple;
#endif
  TFile *hfile = (TFile*)gROOT->FindObject("hsimple.root"); if (hfile) hfile->Close();
  hfile = new TFile("hsimple.root","RECREATE","Demo ROOT file with histograms");
  ntuple = new TNtuple("ntuple","Demo ntuple","px:py:pz:random:i");


  gBenchmark->Start("hsimple");

// Fill histograms randomly
  gRandom->SetSeed();
  Float_t px, py, pz;
  for (Int_t i = 0; i < 2000; i++) {
     gRandom->Rannor(px,py);
     pz = px*px + py*py;
     Float_t random = gRandom->Rndm(1);
     ntuple->Fill(px,py,pz,random,i);
  }
  gBenchmark->Show("hsimple");

  hfile->Write();

#ifdef ClingWorkAroundBrokenUnnamedReturn
   int res = 0;
#else
   return 0;
#endif
}
