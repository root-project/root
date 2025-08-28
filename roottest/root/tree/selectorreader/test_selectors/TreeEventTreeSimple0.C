#define TreeEventTreeSimple0_cxx

#include "generated_selectors/TreeEventTreeSimple0.h"
#include <TH2.h>
#include <TStyle.h>

void TreeEventTreeSimple0::Begin(TTree * /*tree*/)
{
   TString option = GetOption();
}

void TreeEventTreeSimple0::SlaveBegin(TTree * /*tree*/)
{
   TString option = GetOption();
}

Bool_t TreeEventTreeSimple0::Process(Long64_t entry)
{
   fReader.SetEntry(entry);

   fprintf(stderr, "EventSize: %d\n", Event_branch->fEventSize);

   fprintf(stderr, "Px:");
   for(Int_t i = 0; i < Event_branch->fParticles.size(); ++i)
      fprintf(stderr, " %.1lf", Event_branch->fParticles[i].fPosX);
   fprintf(stderr, "\n");

   fprintf(stderr, "Py:");
   for(Int_t i = 0; i < Event_branch->fParticles.size(); ++i)
      fprintf(stderr, " %.1lf", Event_branch->fParticles[i].fPosY);
   fprintf(stderr, "\n");

   fprintf(stderr, "Pz:");
   for(Int_t i = 0; i < Event_branch->fParticles.size(); ++i)
      fprintf(stderr, " %.1lf", Event_branch->fParticles[i].fPosZ);
   fprintf(stderr, "\n");

   return kTRUE;
}

void TreeEventTreeSimple0::SlaveTerminate() { }

void TreeEventTreeSimple0::Terminate() { }
