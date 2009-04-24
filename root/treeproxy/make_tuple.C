#include <TFile.h>
#include <TTree.h>
#include <TRandom3.h>

struct _my_data {
   float val1;
   float val2;
   float val3;
};
typedef struct _my_data my_data_t;

void make_tuple() {
   my_data_t evt;


   TFile *file = TFile::Open("make_tuple.root","recreate");
   TTree *tree = new TTree("data","Example of data-tree");
   tree->Branch("evt",&evt,"val1/F:val2/F:val3/F");

   for (Int_t i=0;i<10;++i) {
      evt.val1 = gRandom->Gaus();
      evt.val2 = gRandom->Uniform();
      evt.val3 = gRandom->PoissonD(3);

      tree->Fill();
   }

   tree->Write();
   file->Close();
}
