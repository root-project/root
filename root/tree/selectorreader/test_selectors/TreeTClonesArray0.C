#define TreeTClonesArray0_cxx

#include "../generated_selectors/TreeTClonesArray0.h"
#include <TH2.h>
#include <TStyle.h>

void TreeTClonesArray0::Begin(TTree * /*tree*/)
{
   TString option = GetOption();
}

void TreeTClonesArray0::SlaveBegin(TTree * /*tree*/)
{
   TString option = GetOption();
}

Bool_t TreeTClonesArray0::Process(Long64_t entry)
{
   fReader.SetEntry(entry);

   for (Int_t i = 0; i < arr.GetSize(); i++) {
      ClassC c = arr[i];
      fprintf(stderr, "%.1f %d ", c.GetPx(), c.GetEv());
   }
   fprintf(stderr, "\n");

   return kTRUE;
}

void TreeTClonesArray0::SlaveTerminate() { }

void TreeTClonesArray0::Terminate(){ }
