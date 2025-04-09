#include "TFile.h"
#include "TTreeReader.h"
#include "TTreeReaderArray.h"

#include "arrayHolder.h"

void colWiseWrite_2() {
   auto file = TFile::Open("file_colWiseWrite_2.root", "RECREATE");
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
