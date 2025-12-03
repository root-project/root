#define TreeClass0_cxx

#include "generated_selectors/TreeClass0.h"
#include <TH2.h>
#include <TStyle.h>

void TreeClass0::Begin(TTree * /*tree*/)
{
   TString option = GetOption();
}

void TreeClass0::SlaveBegin(TTree * /*tree*/)
{
   TString option = GetOption();
}

Bool_t TreeClass0::Process(Long64_t entry)
{
   fReader.SetEntry(entry);

   fprintf(stderr, "Ev: %d, Px: %.1f\n", ClassC_branch->GetEv(), ClassC_branch->GetPx());

   return kTRUE;
}

void TreeClass0::SlaveTerminate() { }

void TreeClass0::Terminate() { }
