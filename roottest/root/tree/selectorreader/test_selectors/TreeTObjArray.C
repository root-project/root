#define TreeTObjArray_cxx

#include "generated_selectors/TreeTObjArray.h"
#include <TH2.h>
#include <TStyle.h>

void TreeTObjArray::Begin(TTree * /*tree*/)
{
   TString option = GetOption();
}

void TreeTObjArray::SlaveBegin(TTree * /*tree*/)
{
   TString option = GetOption();
}

Bool_t TreeTObjArray::Process(Long64_t entry)
{
   fReader.SetEntry(entry);

   for (Int_t i = 0; i < arr->GetSize(); i++) {
      ClassC *c = (ClassC*)((*arr)[i]);
      fprintf(stderr, "%.1f %d ", c->GetPx(), c->GetEv());
   }
   fprintf(stderr, "\n");

   return kTRUE;
}

void TreeTObjArray::SlaveTerminate() { }

void TreeTObjArray::Terminate() { }
