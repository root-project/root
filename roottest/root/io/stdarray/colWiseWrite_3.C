#include "TFile.h"
#include "TTreeReader.h"
#include "TTreeReaderArray.h"
#define ARRAYHOLDER_STDARRAY
#include "arrayHolder.h"

void colWiseWrite_3() {
   auto file = TFile::Open("file_colWiseWrite_3.root", "RECREATE");
   auto tree = new TTree("mytree", "mytree");

   MetaArrayHolder a_metaholder;

   tree->Branch("mybranch", &a_metaholder);
   for (int i=0;i<10;++i) {
      a_metaholder.Set(i,i+1,i+2);
      std::cout << a_metaholder.ToString();
      tree->Fill();
   }
   file->Write();
   file->Close();
}
