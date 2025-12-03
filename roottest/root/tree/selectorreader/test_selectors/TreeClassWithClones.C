#define TreeClassWithClones_cxx

#include "generated_selectors/TreeClassWithClones.h"
#include <TH2.h>
#include <TStyle.h>


void TreeClassWithClones::Begin(TTree * /*tree*/)
{
   TString option = GetOption();
}

void TreeClassWithClones::SlaveBegin(TTree * /*tree*/)
{
   TString option = GetOption();
}

Bool_t TreeClassWithClones::Process(Long64_t entry)
{
   fReader.SetEntry(entry);

   fprintf(stderr, "Px:");
   for (Int_t i = 0; i < arr_fPosX.GetSize(); ++i) fprintf(stderr, " %.1lf", arr_fPosX[i]);
   fprintf(stderr, "\n");

   fprintf(stderr, "Py:");
   for (Int_t i = 0; i < arr_fPosY.GetSize(); ++i) fprintf(stderr, " %.1lf", arr_fPosY[i]);
   fprintf(stderr, "\n");

   fprintf(stderr, "Pz:");
   for (Int_t i = 0; i < arr_fPosZ.GetSize(); ++i) fprintf(stderr, " %.1lf", arr_fPosZ[i]);
   fprintf(stderr, "\n");

   return kTRUE;
}

void TreeClassWithClones::SlaveTerminate() { }

void TreeClassWithClones::Terminate() { }
