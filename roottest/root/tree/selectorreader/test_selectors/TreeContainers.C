#define TreeContainers_cxx

#include "generated_selectors/TreeContainers.h"
#include <TH2.h>
#include <TStyle.h>

void TreeContainers::Begin(TTree * /*tree*/)
{
   TString option = GetOption();
}

void TreeContainers::SlaveBegin(TTree * /*tree*/)
{
   TString option = GetOption();
}

Bool_t TreeContainers::Process(Long64_t entry)
{
   fReader.SetEntry(entry);

   for (Int_t i = 0; i < vectorBranch.GetSize(); ++i) fprintf(stderr, "%d ", vectorBranch[i]);
   fprintf(stderr, "\n");

   for (Int_t i = 0; i < setBranch.GetSize(); ++i) fprintf(stderr, "%d ", setBranch[i]);
   fprintf(stderr, "\n");

   for (Int_t i = 0; i < listBranch.GetSize(); ++i) fprintf(stderr, "%d ", listBranch[i]);
   fprintf(stderr, "\n");

   return kTRUE;
}

void TreeContainers::SlaveTerminate() { }

void TreeContainers::Terminate() { }
