#define TreeNestedVector_cxx

#include "../generated_selectors/TreeNestedVector.h"
#include <TH2.h>
#include <TStyle.h>


void TreeNestedVector::Begin(TTree * /*tree*/)
{
   TString option = GetOption();
}

void TreeNestedVector::SlaveBegin(TTree * /*tree*/)
{
   TString option = GetOption();
}

Bool_t TreeNestedVector::Process(Long64_t entry)
{
   fReader.SetEntry(entry);

   for (Int_t i = 0; i < vec_branch_fEventSize.GetSize(); ++i) {
      fprintf(stderr, "EventSize: %d\n", vec_branch_fEventSize[i]);
      // TODO: this is not working (TTreeReaderArray<vector<T>> is not supproted yet)
      //fprintf(stderr, "Vector size: %zu\n", vec_branch_fParticles[i].size());
      /*for(Int_t j = 0; j < vec_branch_fParticles[i].size(); ++j) {
         fprintf(stderr, "\tParticle: PosX: %.1lf PosY: %.1lf PosZ: %.1lf\n",
                     vec_branch_fParticles[i][j].fPosX,
                     vec_branch_fParticles[i][j].fPosY,
                     vec_branch_fParticles[i][j].fPosZ);
      }*/
   }

   return kTRUE;
}

void TreeNestedVector::SlaveTerminate() { }

void TreeNestedVector::Terminate() { }
