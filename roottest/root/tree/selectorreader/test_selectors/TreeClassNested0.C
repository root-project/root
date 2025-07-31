#define TreeClassNested0_cxx

#include "generated_selectors/TreeClassNested0.h"
#include <TH2.h>
#include <TStyle.h>

void TreeClassNested0::Begin(TTree * /*tree*/)
{
   TString option = GetOption();
}

void TreeClassNested0::SlaveBegin(TTree * /*tree*/)
{
   TString option = GetOption();
}

Bool_t TreeClassNested0::Process(Long64_t entry)
{
   fReader.SetEntry(entry);

   fprintf(stderr, "Ev: %d, Px: %.1f, Py: %.1lf\n", ClassB_branch->GetC().GetEv(), ClassB_branch->GetC().GetPx(), ClassB_branch->GetPy());

   return kTRUE;
}

void TreeClassNested0::SlaveTerminate() { }

void TreeClassNested0::Terminate() { }
