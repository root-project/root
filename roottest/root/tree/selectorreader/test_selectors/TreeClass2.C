#define TreeClass2_cxx

#include "generated_selectors/TreeClass2.h"
#include <TH2.h>
#include <TStyle.h>

void TreeClass2::Begin(TTree * /*tree*/)
{
   TString option = GetOption();
}

void TreeClass2::SlaveBegin(TTree * /*tree*/)
{
   TString option = GetOption();
}

Bool_t TreeClass2::Process(Long64_t entry)
{
   fReader.SetEntry(entry);

   fprintf(stderr, "Ev: %d, Px: %.1f\n", *fEv, *fPx);

   return kTRUE;
}

void TreeClass2::SlaveTerminate() { }

void TreeClass2::Terminate() { }
