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

// Create a new canvas.
  c1 = new TCanvas("c1","Dynamic Filling Example",200,10,700,500);
  c1->SetFillColor(42);
  c1->GetFrame()->SetFillColor(21);
  c1->GetFrame()->SetBorderSize(6);
  c1->GetFrame()->SetBorderMode(-1);

// Create a new ROOT binary machine independent file.
// Note that this file may contain any kind of ROOT objects, histograms,
// pictures, graphics objects, detector geometries, tracks, events, etc..
// This file is now becoming the current directory.

  TFile *hfile = (TFile*)gROOT->FindObject("hsimple.root"); if (hfile) hfile->Close();
  hfile = new TFile("hsimple.root","RECREATE","Demo ROOT file with histograms");

// Create some histograms, a profile histogram and an ntuple
  hpx    = new TH1F("hpx","This is the px distribution",100,-4,4);
  hpxpy  = new TH2F("hpxpy","py vs px",40,-4,4,40,-4,4);
  hprof  = new TProfile("hprof","Profile of pz versus px",100,-4,4,0,20);
  ntuple = new TNtuple("ntuple","Demo ntuple","px:py:pz:random:i");

//  Set canvas/frame attributes (save old attributes)
  hpx->SetFillColor(48);

  gBenchmark->Start("hsimple");

// Fill histograms randomly
  gRandom->SetSeed();
  Float_t px, py, pz;
  const Int_t kUPDATE = 1000;
  for (Int_t i = 0; i < 25000; i++) {
     gRandom->Rannor(px,py);
     pz = px*px + py*py;
     Float_t random = gRandom->Rndm(1);
     hpx->Fill(px);
     hpxpy->Fill(px,py);
     hprof->Fill(px,pz);
     ntuple->Fill(px,py,pz,random,i);
     if (i && (i%kUPDATE) == 0) {
        if (i == kUPDATE) hpx->Draw();
        c1->Modified();
        c1->Update();
        if (gSystem->ProcessEvents())
           break;
     }
  }
  gBenchmark->Show("hsimple");

// Save all objects in this file
  hpx->SetFillColor(0);
  hfile->Write();
  hpx->SetFillColor(48);
  c1->Modified();
  
// Note that the file is automatically close when application terminates
// or when the file destructor is called.
}
