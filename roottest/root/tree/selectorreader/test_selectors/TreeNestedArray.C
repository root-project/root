#define TreeNestedArray_cxx

#include "../generated_selectors/TreeNestedArray.h"
#include <TH2.h>
#include <TStyle.h>


void TreeNestedArray::Begin(TTree * /*tree*/)
{
   TString option = GetOption();
}

void TreeNestedArray::SlaveBegin(TTree * /*tree*/)
{
   TString option = GetOption();
}

Bool_t TreeNestedArray::Process(Long64_t entry)
{
   fReader.SetEntry(entry);

   return kTRUE;
}

void TreeNestedArray::SlaveTerminate() { }

void TreeNestedArray::Terminate() { }
