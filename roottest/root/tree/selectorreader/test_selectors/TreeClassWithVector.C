#define TreeClassWithVector_cxx

#include "../generated_selectors/TreeClassWithVector.h"
#include <TH2.h>
#include <TStyle.h>


void TreeClassWithVector::Begin(TTree * /*tree*/)
{
   TString option = GetOption();
}

void TreeClassWithVector::SlaveBegin(TTree * /*tree*/)
{
   TString option = GetOption();
}

Bool_t TreeClassWithVector::Process(Long64_t entry)
{
   fReader.SetEntry(entry);

   for (Int_t i = 0; i < vec.GetSize(); ++i) fprintf(stderr, "%d ", vec[i]);
   fprintf(stderr, "\n");

   for (Int_t i = 0; i < vecBool->size(); ++i) fprintf(stderr, "%d ", static_cast<int>((*vecBool)[i]));
   fprintf(stderr, "\n");

   return kTRUE;
}

void TreeClassWithVector::SlaveTerminate() { }

void TreeClassWithVector::Terminate() { }
