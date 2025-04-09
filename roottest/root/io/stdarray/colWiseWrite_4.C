#include "TFile.h"
#include "TTreeReader.h"
#include "TTreeReaderArray.h"

#include "arrayHolder.h"

void colWiseWrite_4() {
   auto file = TFile::Open("file_colWiseWrite_4.root", "RECREATE");
   auto tree = new TTree("mytree", "mytree");

   MetaArrayHolder a_metaholder;

   tree->Branch("mybranch", &a_metaholder);
   for (int i=0;i<10;++i) {
      a_metaholder.Set(i,i+1,i+2);
      tree->Fill();
   }
   file->Write();
   file->Close();
}
