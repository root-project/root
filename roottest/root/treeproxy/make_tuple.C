#include <TFile.h>
#include <TTree.h>
#include <TRandom3.h>

struct _my_data {
   float val1;
   float val2;
   float val3;
};
typedef struct _my_data my_data_t;

struct my_array {
   int   n;
   float arr[10];
};

void make_tuple() {
   my_data_t evt;
   my_array trk;
   trk.n = 10;

   TFile *file = TFile::Open("make_tuple.root","recreate");
   TTree *tree = new TTree("data","Example of data-tree");
   tree->Branch("evt",&evt,"val1/F:val2/F:val3/F");
   tree->Branch("trk",&trk,"N/I:arr[N]/F");

   for (Int_t i=0;i<10;++i) {
      evt.val1 = gRandom->Gaus();
      evt.val2 = gRandom->Uniform();
      evt.val3 = gRandom->PoissonD(3);
      trk.n = i;
      for(Int_t t=0;t<i;++t) {
         trk.arr[t] = i*100+t;
      }
      tree->Fill();
   }

   tree->Write();
   file->Close();
}
