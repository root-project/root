#define TreeClassNested2_fC_cxx

#include "../generated_selectors/TreeClassNested2_fC.h"
#include <TH2.h>
#include <TStyle.h>


void TreeClassNested2_fC::Begin(TTree * /*tree*/)
{
   TString option = GetOption();
}

void TreeClassNested2_fC::SlaveBegin(TTree * /*tree*/)
{
   TString option = GetOption();
}

Bool_t TreeClassNested2_fC::Process(Long64_t entry)
{
   fReader.SetEntry(entry);

   fprintf(stderr, "Ev: %d Px: %.1f\n", *fC_fEv, *fC_fPx);

   return kTRUE;
}

void TreeClassNested2_fC::SlaveTerminate() { }

void TreeClassNested2_fC::Terminate() { }
