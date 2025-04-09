#define TreeClassNested2_atfC_cxx

#include "../generated_selectors/TreeClassNested2_atfC.h"
#include <TH2.h>
#include <TStyle.h>


void TreeClassNested2_atfC::Begin(TTree * /*tree*/)
{
   TString option = GetOption();
}

void TreeClassNested2_atfC::SlaveBegin(TTree * /*tree*/)
{
   TString option = GetOption();
}

Bool_t TreeClassNested2_atfC::Process(Long64_t entry)
{
   fReader.SetEntry(entry);

   fprintf(stderr, "Ev: %d Px: %.1f\n", fC->GetEv(), fC->GetPx());

   return kTRUE;
}

void TreeClassNested2_atfC::SlaveTerminate() { }

void TreeClassNested2_atfC::Terminate() { }
