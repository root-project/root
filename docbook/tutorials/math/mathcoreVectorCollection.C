//
// Example showing how to write and read a std vector of ROOT::Math LorentzVector in a ROOT tree. 
// In the write() function a variable number of track Vectors is generated 
// according to a Poisson distribution with random momentum uniformly distributed 
// in phi and eta. 
// In the read() the vectors are read back and the content analyzed and 
// some information such as number of tracks per event or the track pt 
// distributions are displayed  in a canvas. 
//
// To execute the macro type in: 
//
//   root[0]: .x  mathcoreVectorCollection.C
//Author: Andras Zsenei     



#include "TRandom.h"
#include "TStopwatch.h"
#include "TSystem.h"
#include "TFile.h"
#include "TTree.h"
#include "TH1D.h"
#include "TCanvas.h"
#include "TMath.h"

#include <iostream>

// CINT does not understand some files included by LorentzVector
#include "Math/Vector3D.h"
#include "Math/Vector4D.h"

using namespace ROOT::Math;



double write(int n) { 



  TRandom R; 
  TStopwatch timer;


  TFile f1("mathcoreLV.root","RECREATE");

  // create tree
  TTree t1("t1","Tree with new LorentzVector");

  std::vector<ROOT::Math::XYZTVector>  tracks; 
  std::vector<ROOT::Math::XYZTVector> * pTracks = &tracks; 
  t1.Branch("tracks","std::vector<ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > >",&pTracks);

  double M = 0.13957;  // set pi+ mass

  timer.Start();
  double sum = 0;
  for (int i = 0; i < n; ++i) { 
    int nPart = R.Poisson(5);
    pTracks->clear();
    pTracks->reserve(nPart); 
    for (int j = 0; j < nPart; ++j) {
      double px = R.Gaus(0,10);
      double py = R.Gaus(0,10);
      double pt = sqrt(px*px +py*py);
      double eta = R.Uniform(-3,3);
      double phi = R.Uniform(0.0 , 2*TMath::Pi() );
      RhoEtaPhiVector vcyl( pt, eta, phi); 
      // set energy 
      double E = sqrt( vcyl.R()*vcyl.R() + M*M);  
      XYZTVector q( vcyl.X(), vcyl.Y(), vcyl.Z(), E);
      // fill track vector
      pTracks->push_back(q);
      // evaluate sum of components to check 
      sum += q.x()+q.y()+q.z()+q.t();
    }
    t1.Fill(); 
  }

  f1.Write();
  timer.Stop();
  std::cout << " Time for new Vector " << timer.RealTime() << "  " << timer.CpuTime() << std::endl; 

  t1.Print();
  return sum;
}



double read() { 


  TRandom R; 
  TStopwatch timer;


  TH1D * h1 = new TH1D("h1","total event  energy ",100,0,1000.);
  TH1D * h2 = new TH1D("h2","Number of track per event",21,-0.5,20.5);
  TH1D * h3 = new TH1D("h3","Track Energy",100,0,200);
  TH1D * h4 = new TH1D("h4","Track Pt",100,0,100);
  TH1D * h5 = new TH1D("h5","Track Eta",100,-5,5);
  TH1D * h6 = new TH1D("h6","Track Cos(theta)",100,-1,1);


  TFile f1("mathcoreLV.root");

  // create tree
  TTree *t1 = (TTree*)f1.Get("t1");

  std::vector<ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > > * pTracks = 0;
  t1->SetBranchAddress("tracks",&pTracks);

  timer.Start();
  int n = (int) t1->GetEntries();
  std::cout << " Tree Entries " << n << std::endl; 
  double sum=0;
  for (int i = 0; i < n; ++i) { 
    t1->GetEntry(i);
    int ntrk = pTracks->size(); 
    h3->Fill(ntrk);
    XYZTVector q; 
    for (int j = 0; j < ntrk; ++j) { 
      XYZTVector v = (*pTracks)[j]; 
      q += v; 
      h3->Fill(v.E());
      h4->Fill(v.Pt());
      h5->Fill(v.Eta());
      h6->Fill(cos(v.Theta()));
      sum += v.x() + v.y() + v.z() + v.t();
    }
    h1->Fill(q.E() );
    h2->Fill(ntrk);
  }


  timer.Stop();
  std::cout << " Time for new Vector " << timer.RealTime() << "  " << timer.CpuTime() << std::endl; 


  
  TCanvas *c1 = new TCanvas("c1","demo of Trees",10,10,600,800);
  c1->Divide(2,3); 
  
  c1->cd(1);
  h1->Draw();
  c1->cd(2);
  h2->Draw();
  c1->cd(3);
  h3->Draw();
  c1->cd(3);
  h3->Draw();
  c1->cd(4);
  h4->Draw();
  c1->cd(5);
  h5->Draw();
  c1->cd(6);
  h6->Draw();

  return sum;
}



int mathcoreVectorCollection() { 


#if defined(__CINT__) && !defined(__MAKECINT__) 

  gSystem->Load("libMathCore");  
  gSystem->Load("libPhysics");  
  // in CINT need to do that after having loading the library
  using namespace ROOT::Math;

  cout << "This tutorial can run only using ACliC, compiling it by doing: " << endl;
  cout << "\t  .x tutorials/math/mathcoreVectorCollection.C+" << endl; 
  //gROOT->ProcessLine(".x tutorials/math/mathcoreVectorCollection.C+"); 
  return 0;

#else

  
  int nEvents = 10000;

  double s1 = write(nEvents);

  double s2 = read();

  if (fabs(s1-s2) > s1*1.E-15 ) { 
    std::cout << "ERROR: Found difference in Vector when reading  ( " << s1 << " != " << s2 << " diff = " << fabs(s1-s2) << " ) " << std::endl;
    return -1;
  }
  return 0;

#endif
}


#ifndef __CINT__
int main() { 
  return mathcoreVectorCollection(); 
}
#endif

