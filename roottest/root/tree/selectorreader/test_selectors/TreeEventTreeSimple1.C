#define TreeEventTreeSimple1_cxx

#include "../generated_selectors/TreeEventTreeSimple1.h"
#include <TH2.h>
#include <TStyle.h>

void TreeEventTreeSimple1::Begin(TTree * /*tree*/)
{
   TString option = GetOption();
}

void TreeEventTreeSimple1::SlaveBegin(TTree * /*tree*/)
{
   TString option = GetOption();
}

Bool_t TreeEventTreeSimple1::Process(Long64_t entry)
{
   fReader.SetEntry(entry);

   fprintf(stderr, "EventSize: %d\n", *fEventSize);
   
   fprintf(stderr, "Px:");
   for(Int_t i = 0; i < fParticles.GetSize(); ++i)
      fprintf(stderr, " %.1lf", fParticles[i].fPosX);
   fprintf(stderr, "\n");

   fprintf(stderr, "Py:");
   for(Int_t i = 0; i < fParticles.GetSize(); ++i)
      fprintf(stderr, " %.1lf", fParticles[i].fPosY);
   fprintf(stderr, "\n");

   fprintf(stderr, "Pz:");
   for(Int_t i = 0; i < fParticles.GetSize(); ++i)
      fprintf(stderr, " %.1lf", fParticles[i].fPosZ);
   fprintf(stderr, "\n");

   return kTRUE;
}

void TreeEventTreeSimple1::SlaveTerminate() { }

void TreeEventTreeSimple1::Terminate() { }
