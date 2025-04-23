#define TreeDuplicateName_cxx

#include "../generated_selectors/TreeDuplicateName.h"
#include <TH2.h>
#include <TStyle.h>

void TreeDuplicateName::Begin(TTree * /*tree*/)
{
   TString option = GetOption();
}

void TreeDuplicateName::SlaveBegin(TTree * /*tree*/)
{
   TString option = GetOption();
}

Bool_t TreeDuplicateName::Process(Long64_t entry)
{
   fReader.SetEntry(entry);

   fprintf(stderr, "Ev: %d, Px: %.1f\n", *fEv, *fPx);

   return kTRUE;
}

void TreeDuplicateName::SlaveTerminate() { }

void TreeDuplicateName::Terminate() { }
