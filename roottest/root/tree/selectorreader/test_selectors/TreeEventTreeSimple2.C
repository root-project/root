#define TreeEventTreeSimple2_cxx

#include "generated_selectors/TreeEventTreeSimple2.h"
#include <TH2.h>
#include <TStyle.h>

void TreeEventTreeSimple2::Begin(TTree * /*tree*/)
{
   TString option = GetOption();
}

void TreeEventTreeSimple2::SlaveBegin(TTree * /*tree*/)
{
   TString option = GetOption();
}

Bool_t TreeEventTreeSimple2::Process(Long64_t entry)
{
   fReader.SetEntry(entry);

   fprintf(stderr, "EventSize: %d\n", *fEventSize);

   fprintf(stderr, "Px:");
   for(Int_t i = 0; i < fParticles_fPosX.GetSize(); ++i)
      fprintf(stderr, " %.1lf", fParticles_fPosX[i]);
   fprintf(stderr, "\n");

   fprintf(stderr, "Py:");
   for(Int_t i = 0; i < fParticles_fPosY.GetSize(); ++i)
      fprintf(stderr, " %.1lf", fParticles_fPosY[i]);
   fprintf(stderr, "\n");

   fprintf(stderr, "Pz:");
   for(Int_t i = 0; i < fParticles_fPosZ.GetSize(); ++i)
      fprintf(stderr, " %.1lf", fParticles_fPosZ[i]);
   fprintf(stderr, "\n");

   return kTRUE;
}

void TreeEventTreeSimple2::SlaveTerminate() { }

void TreeEventTreeSimple2::Terminate() { }
