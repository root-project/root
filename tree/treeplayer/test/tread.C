#include "TFile.h"
#include "TTreeReader.h"
#include "TInterpreter.h"
#include "TSystem.h"
#include "MyParticle.h"
#include "TTreeReaderValue.h"
#include "TTreeReaderArray.h"
#include <vector>

#ifdef __CINT__
#pragma link C++ class TTreeReaderValue<std::vector<MyParticle*>>+;
#pragma link C++ class TTreeReaderValue<MyParticle>+;
#pragma link C++ class TTreeReaderArray<double>+;
#endif

void tread_obj() {
   // Reading object branches:
   TFile* f = TFile::Open("tr.root");
   TTreeReader tr("T");
   TTreeReaderValue< MyParticle > p(tr, "p");
   while (tr.SetNextEntry()) {
      printf("Particle momentum: %g\n", *p->P());
   }
   delete f;
}

void tread_makeclass() {
   // Reading members stored in a collection ("makeclass mode")
   TFile* f = TFile::Open("tr.root");
   TTreeReader tr("T");
   TTreeReaderArray<double> e(tr, "v.fPos.fY");
   while (tr.SetNextEntry()) {
      if (!e.IsEmpty())
         printf("lead muon energy: %g\n", e.At(0));
   }
   delete f;
}

void tread() {
   gSystem->Load("libTreeReader_C");
   tread_obj();
   //tread_makeclass();
}
