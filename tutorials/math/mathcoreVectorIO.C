//
// Example of  I/O of a mathcore Lorentz Vectors in a Tree and comparison with a TLorentzVector
// A ROOT tree is written and read in both using either a XYZTVector or /// a TLorentzVector. 
// 
//  To execute the macro type in: 
//
// root[0]: .x  mathcoreVectorIO.C
//Author: Lorenzo Moneta


#include "TRandom2.h"
#include "TStopwatch.h"
#include "TSystem.h"
#include "TFile.h"
#include "TTree.h"
#include "TH1D.h"
#include "TCanvas.h"

#include <iostream>

#include "TLorentzVector.h"

#include "Math/Vector4D.h"

using namespace ROOT::Math;

 


void write(int n) { 


  TRandom2 R; 
  TStopwatch timer;


  R.SetSeed(1);
  timer.Start();
  double s = 0; 
  for (int i = 0; i < n; ++i) { 
        s  += R.Gaus(0,10);
	s  += R.Gaus(0,10);
	s  += R.Gaus(0,10);
	s  += R.Gaus(100,10);
  }

  timer.Stop();
  std::cout << s/double(n) << std::endl;
  std::cout << " Time for Random gen " << timer.RealTime() << "  " << timer.CpuTime() << std::endl; 
  

  TFile f1("mathcoreVectorIO_1.root","RECREATE");

  // create tree
  TTree t1("t1","Tree with new LorentzVector");

  XYZTVector *v1 = new XYZTVector();
  t1.Branch("LV branch","ROOT::Math::XYZTVector",&v1);

  R.SetSeed(1);
  timer.Start();
  for (int i = 0; i < n; ++i) { 
        double Px = R.Gaus(0,10);
	double Py = R.Gaus(0,10);
	double Pz = R.Gaus(0,10);
	double E  = R.Gaus(100,10);
	//CylindricalEta4D<double> & c = v1->Coordinates();
	//c.SetValues(Px,pY,pZ,E);
	v1->SetCoordinates(Px,Py,Pz,E);
	t1.Fill(); 
  }

  f1.Write();
  timer.Stop();
  std::cout << " Time for new Vector " << timer.RealTime() << "  " << timer.CpuTime() << std::endl; 

  t1.Print();

  // create tree with old LV 

  TFile f2("mathcoreVectorIO_2.root","RECREATE");
  TTree t2("t2","Tree with TLorentzVector");

  TLorentzVector * v2 = new TLorentzVector();
  TLorentzVector::Class()->IgnoreTObjectStreamer();
  TVector3::Class()->IgnoreTObjectStreamer();

  t2.Branch("TLV branch","TLorentzVector",&v2,16000,2);

  R.SetSeed(1);
  timer.Start();
  for (int i = 0; i < n; ++i) { 
        double Px = R.Gaus(0,10);
	double Py = R.Gaus(0,10);
	double Pz = R.Gaus(0,10);
	double E  = R.Gaus(100,10);
	v2->SetPxPyPzE(Px,Py,Pz,E);
	t2.Fill(); 
  }

  f2.Write();
  timer.Stop();
  std::cout << " Time for old Vector " << timer.RealTime() << "  " << timer.CpuTime() << endl; 

  t2.Print();
}


void read() { 




  TRandom R; 
  TStopwatch timer;



  TFile f1("mathcoreVectorIO_1.root");

  // create tree
  TTree *t1 = (TTree*)f1.Get("t1");

  XYZTVector *v1 = 0;
  t1->SetBranchAddress("LV branch",&v1);

  timer.Start();
  int n = (int) t1->GetEntries();
  std::cout << " Tree Entries " << n << std::endl; 
  double etot=0;
  for (int i = 0; i < n; ++i) { 
    t1->GetEntry(i);
    etot += v1->Px();
    etot += v1->Py();
    etot += v1->Pz();
    etot += v1->E();
  }


  timer.Stop();
  std::cout << " Time for new Vector " << timer.RealTime() << "  " << timer.CpuTime() << std::endl; 

  std::cout << " TOT average : n = " << n << "\t " << etot/double(n) << endl; 


  // create tree with old LV 

  TFile f2("mathcoreVectorIO_2.root");
  TTree *t2 = (TTree*)f2.Get("t2");


  TLorentzVector * v2 = 0;
  t2->SetBranchAddress("TLV branch",&v2);

  timer.Start();
  n = (int) t2->GetEntries();
  std::cout << " Tree Entries " << n << std::endl; 
  etot = 0;
  for (int i = 0; i < n; ++i) { 
    t2->GetEntry(i);
    etot  += v2->Px();
    etot  += v2->Py();
    etot  += v2->Pz();
    etot  += v2->E();
  }

  timer.Stop();
  std::cout << " Time for old Vector " << timer.RealTime() << "  " << timer.CpuTime() << endl; 
  std::cout << " TOT average:\t" << etot/double(n) << endl; 
}



void mathcoreVectorIO() { 


#if defined(__CINT__) && !defined(__MAKECINT__) 

  gSystem->Load("libMathCore");  
  gSystem->Load("libPhysics");  
  // in CINT need to do that after having loading the library
  using namespace ROOT::Math;

  cout << "This tutorial can run only using ACliC, compiling it by doing: " << endl;
  cout << "\t  .x tutorials/math/mathcoreVectorCollection.C+" << endl; 
  //gROOT->ProcessLine(".x tutorials/math/mathcoreVectorCollection.C+"); 
  return;

#endif

  
  int nEvents = 100000;

  write(nEvents);

  read();
}
  
