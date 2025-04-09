#define TreeStruct_cxx

#include "../generated_selectors/TreeStruct.h"
#include <TH2.h>
#include <TStyle.h>

void TreeStruct::Begin(TTree * /*tree*/)
{
   TString option = GetOption();
}

void TreeStruct::SlaveBegin(TTree * /*tree*/)
{
   TString option = GetOption();
}

Bool_t TreeStruct::Process(Long64_t entry)
{
   fReader.SetEntry(entry);

   fprintf(stderr, "%.1f %.1f %.1f %d\n", *px, *py, *py, *ev);

   return kTRUE;
}

void TreeStruct::SlaveTerminate() { }

void TreeStruct::Terminate() { }
