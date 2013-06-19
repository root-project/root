#include "TFile.h"
#include "TTreeReader.h"
#include "TInterpreter.h"
#include "TSystem.h"
#include "MyParticle.h"
#include "TTreeReaderValue.h"
#include <vector>

#ifdef __CINT__
#pragma link C++ class TTreeReaderValuePtr<std::vector<MyParticle*>>+;
#pragma link C++ class TTreeReaderValuePtr<MyParticle>+;
#pragma link C++ class TTreeReaderArray<double>+;
#endif

void tread_obj() {
   // Reading object branches:
   TFile* f = TFile::Open("tr.root");
   TTreeReader tr("T");
   TTreeReaderValuePtr< MyParticle > p(tr, "p");
   while (tr.GetNextEntry()) {
      printf("Particle momentum: %g\n", p->GetP());
   }
   delete f;
}

void tread_makeclass() {
   // Reading members stored in a collection ("makeclass mode")
   TFile* f = TFile::Open("tr.root");
   TTreeReader tr("T");
   TTreeReaderArray<double> e(tr, "v.fPos.fY");
   while (tr.GetNextEntry()) {
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
