#define TreeClassWithArray_cxx

#include "generated_selectors/TreeClassWithArray.h"
#include <TH2.h>
#include <TStyle.h>


void TreeClassWithArray::Begin(TTree * /*tree*/)
{
   TString option = GetOption();
}

void TreeClassWithArray::SlaveBegin(TTree * /*tree*/)
{
   TString option = GetOption();
}

Bool_t TreeClassWithArray::Process(Long64_t entry)
{
   fReader.SetEntry(entry);

   for (Int_t i = 0; i < arr.GetSize(); ++i) fprintf(stderr, "%d ", arr[i]);
   fprintf(stderr, "\n");

   return kTRUE;
}

void TreeClassWithArray::SlaveTerminate() { }

void TreeClassWithArray::Terminate() { }
