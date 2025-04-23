#define TreeClassNested2_empty_cxx

#include "../generated_selectors/TreeClassNested2_empty.h"
#include <TH2.h>
#include <TStyle.h>


void TreeClassNested2_empty::Begin(TTree * /*tree*/)
{
   TString option = GetOption();
}

void TreeClassNested2_empty::SlaveBegin(TTree * /*tree*/)
{
   TString option = GetOption();
}

Bool_t TreeClassNested2_empty::Process(Long64_t entry)
{
   fReader.SetEntry(entry);

   fprintf(stderr, "Ev: %d Px: %.1f Py: %.1f\n", *fC_fEv, *fC_fPx, *fPy);

   return kTRUE;
}

void TreeClassNested2_empty::SlaveTerminate() { }

void TreeClassNested2_empty::Terminate() { }
