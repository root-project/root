#define TreeTClonesArray2_cxx

#include "generated_selectors/TreeTClonesArray2.h"
#include <TH2.h>
#include <TStyle.h>

void TreeTClonesArray2::Begin(TTree * /*tree*/)
{
   TString option = GetOption();
}

void TreeTClonesArray2::SlaveBegin(TTree * /*tree*/)
{
   TString option = GetOption();
}

Bool_t TreeTClonesArray2::Process(Long64_t entry)
{
   fReader.SetEntry(entry);

   fprintf(stderr, "fEv:");
   for (Int_t i = 0; i < arr_fEv.GetSize(); ++i) fprintf(stderr, " %d", arr_fEv[i]);
   fprintf(stderr, "\n");

   fprintf(stderr, "fPx:");
   for (Int_t i = 0; i < arr_fPx.GetSize(); ++i) fprintf(stderr, "% .1f", arr_fPx[i]);
   fprintf(stderr, "\n");

   return kTRUE;
}

void TreeTClonesArray2::SlaveTerminate() { }

void TreeTClonesArray2::Terminate() { }
