#define TreeClassNested2_alltop_cxx

#include "generated_selectors/TreeClassNested2_alltop.h"
#include <TH2.h>
#include <TStyle.h>


void TreeClassNested2_alltop::Begin(TTree * /*tree*/)
{
   TString option = GetOption();
}

void TreeClassNested2_alltop::SlaveBegin(TTree * /*tree*/)
{
   TString option = GetOption();
}

Bool_t TreeClassNested2_alltop::Process(Long64_t entry)
{
   fReader.SetEntry(entry);

   fprintf(stderr, "Ev: %d Px: %.1f Py: %.1f\n", ClassB_branch->GetC().GetEv(), ClassB_branch->GetC().GetPx(), ClassB_branch->GetPy());

   return kTRUE;
}

void TreeClassNested2_alltop::SlaveTerminate() { }

void TreeClassNested2_alltop::Terminate() { }
