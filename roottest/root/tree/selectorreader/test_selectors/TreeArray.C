#define TreeArray_cxx

#include "generated_selectors/TreeArray.h"
#include <TH2.h>
#include <TStyle.h>

void TreeArray::Begin(TTree * /*tree*/)
{
   TString option = GetOption();
}

void TreeArray::SlaveBegin(TTree * /*tree*/)
{
   TString option = GetOption();
}

Bool_t TreeArray::Process(Long64_t entry)
{
   fReader.SetEntry(entry);

   for (Int_t i = 0; i < arr.GetSize(); ++i) fprintf(stderr, "%.1f ", arr[i]);
   fprintf(stderr, "\n");

   return kTRUE;
}

void TreeArray::SlaveTerminate() { }

void TreeArray::Terminate() { }
