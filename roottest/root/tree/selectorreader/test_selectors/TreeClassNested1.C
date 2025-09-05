#define TreeClassNested1_cxx

#include "generated_selectors/TreeClassNested1.h"
#include <TH2.h>
#include <TStyle.h>

void TreeClassNested1::Begin(TTree * /*tree*/)
{
   TString option = GetOption();
}

void TreeClassNested1::SlaveBegin(TTree * /*tree*/)
{
   TString option = GetOption();
}

Bool_t TreeClassNested1::Process(Long64_t entry)
{
   fReader.SetEntry(entry);

   fprintf(stderr, "Ev: %d, Px: %.1f, Py: %.1lf\n", fC->GetEv(), fC->GetPx(), *fPy);

   return kTRUE;
}

void TreeClassNested1::SlaveTerminate() { }

void TreeClassNested1::Terminate() { }
