#define TreeVectorClass2_cxx

#include "generated_selectors/TreeVectorClass2.h"
#include <TH2.h>
#include <TStyle.h>


void TreeVectorClass2::Begin(TTree * /*tree*/)
{
   TString option = GetOption();
}

void TreeVectorClass2::SlaveBegin(TTree * /*tree*/)
{
   TString option = GetOption();
}

Bool_t TreeVectorClass2::Process(Long64_t entry)
{
   fReader.SetEntry(entry);

   fprintf(stderr, "PosX: ");
   for (Int_t i = 0; i < vp_fPosX.GetSize(); ++i) fprintf(stderr, " %.1lf", vp_fPosX[i]);
   fprintf(stderr, "\n");

   fprintf(stderr, "PosY: ");
   for (Int_t i = 0; i < vp_fPosX.GetSize(); ++i) fprintf(stderr, " %.1lf", vp_fPosY[i]);
   fprintf(stderr, "\n");

   fprintf(stderr, "PosZ: ");
   for (Int_t i = 0; i < vp_fPosX.GetSize(); ++i) fprintf(stderr, " %.1lf", vp_fPosZ[i]);
   fprintf(stderr, "\n");
   return kTRUE;
}

void TreeVectorClass2::SlaveTerminate() { }

void TreeVectorClass2::Terminate() { }
