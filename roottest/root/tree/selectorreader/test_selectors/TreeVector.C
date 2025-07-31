#define TreeVector_cxx

#include "generated_selectors/TreeVector.h"
#include <TH2.h>
#include <TStyle.h>

void TreeVector::Begin(TTree * /*tree*/)
{
   TString option = GetOption();
}

void TreeVector::SlaveBegin(TTree * /*tree*/)
{
   TString option = GetOption();
}

Bool_t TreeVector::Process(Long64_t entry)
{
   fReader.SetEntry(entry);

   for (Int_t i = 0; i < vpx.GetSize(); ++i) fprintf(stderr, "%.1f ", vpx[i]);
   fprintf(stderr, "\n");

   for (Int_t i = 0; i < vb->size(); ++i) fprintf(stderr, "%d ", static_cast<int>((*vb)[i]));
   fprintf(stderr, "\n");

   return kTRUE;
}

void TreeVector::SlaveTerminate() { }

void TreeVector::Terminate() { }
