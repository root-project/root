/// \file
/// \ingroup tutorial_math
/// \notebook -nodraw
/// Macro illustrating  I/O with Lorentz Vectors of floats
///
/// \macro_code
///
/// \author Lorenzo Moneta

#include "TRandom.h"
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

   TRandom R;
   TStopwatch timer;

   TFile f1("mathcoreVectorIO_F.root","RECREATE");

   // create tree
   TTree t1("t1","Tree with new Float LorentzVector");

   XYZTVectorF *v1 = new XYZTVectorF();
   t1.Branch("LV branch","ROOT::Math::XYZTVectorF",&v1);

   timer.Start();
   for (int i = 0; i < n; ++i) {
      double Px = R.Gaus(0,10);
      double Py = R.Gaus(0,10);
      double Pz = R.Gaus(0,10);
      double E  = R.Gaus(100,10);
      v1->SetCoordinates(Px,Py,Pz,E);
      t1.Fill();
   }

   f1.Write();
   timer.Stop();
   std::cout << " Time for new Float Vector " << timer.RealTime() << "  " << timer.CpuTime() << std::endl;
   t1.Print();
}

void read() {

   TRandom R;
   TStopwatch timer;

   TFile f1("mathcoreVectorIO_F.root");

   // create tree
   TTree *t1 = (TTree*)f1.Get("t1");

   XYZTVectorF *v1 = 0;
   t1->SetBranchAddress("LV branch",&v1);

   timer.Start();
   int n = (int) t1->GetEntries();
   std::cout << " Tree Entries " << n << std::endl;
   double etot=0;
   for (int i = 0; i < n; ++i) {
      t1->GetEntry(i);
      etot += v1->E();
   }

   timer.Stop();
   std::cout << " Time for new Float Vector " << timer.RealTime() << "  " << timer.CpuTime() << std::endl;
   std::cout << " E average" << n<< "  " << etot << "  " << etot/double(n) << endl;
}

void runIt() {
   int nEvents = 100000;
   write(nEvents);
   read();
}

void mathcoreVectorFloatIO() {
   runIt();

}
