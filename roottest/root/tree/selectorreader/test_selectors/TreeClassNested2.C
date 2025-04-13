#define TreeClassNested2_cxx

#include "../generated_selectors/TreeClassNested2.h"
#include <TH2.h>
#include <TStyle.h>

void TreeClassNested2::Begin(TTree * /*tree*/)
{
   TString option = GetOption();
}

void TreeClassNested2::SlaveBegin(TTree * /*tree*/)
{
   TString option = GetOption();
}

Bool_t TreeClassNested2::Process(Long64_t entry)
{
   fReader.SetEntry(entry);

   fprintf(stderr, "Ev: %d, Px: %.1f, Py: %.1lf\n", *fC_fEv, *fC_fPx, *fPy);

   return kTRUE;
}

void TreeClassNested2::SlaveTerminate() { }

void TreeClassNested2::Terminate() { }
