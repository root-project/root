#define TreeVectorClass0_cxx

#include "../generated_selectors/TreeVectorClass0.h"
#include <TH2.h>
#include <TStyle.h>


void TreeVectorClass0::Begin(TTree * /*tree*/)
{
   TString option = GetOption();
}

void TreeVectorClass0::SlaveBegin(TTree * /*tree*/)
{
   TString option = GetOption();
}

Bool_t TreeVectorClass0::Process(Long64_t entry)
{
   fReader.SetEntry(entry);

   fprintf(stderr, "PosX: ");
   for (Int_t i = 0; i < vp.GetSize(); ++i) fprintf(stderr, " %.1lf", vp[i].fPosX);
   fprintf(stderr, "\n");

   fprintf(stderr, "PosY: ");
   for (Int_t i = 0; i < vp.GetSize(); ++i) fprintf(stderr, " %.1lf", vp[i].fPosY);
   fprintf(stderr, "\n");

   fprintf(stderr, "PosZ: ");
   for (Int_t i = 0; i < vp.GetSize(); ++i) fprintf(stderr, " %.1lf", vp[i].fPosZ);
   fprintf(stderr, "\n");
   
   return kTRUE;
}

void TreeVectorClass0::SlaveTerminate() { }

void TreeVectorClass0::Terminate() { }
