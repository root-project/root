#include "TFile.h"
#include "TTreeReader.h"
#include "TTreeReaderArray.h"
#define ARRAYHOLDER_STDARRAY
#include "arrayHolder.h"

void colWiseWrite_1() {
   auto file = TFile::Open("file_colWiseWrite_1.root", "RECREATE");
   auto tree = new TTree("mytree", "mytree");

   ArrayHolder a_holder;

   tree->Branch("mybranch", &a_holder);
   for (int i=0;i<10;++i) {
      a_holder.Set(i,i+1,i+2);
      tree->Fill();
   }
   file->Write();
   file->Close();
}
