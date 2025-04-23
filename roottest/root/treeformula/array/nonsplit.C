#include "TFile.h"
#include "TTree.h"
#include <vector>

#ifdef __CINT__
#pragma link C++ class std::vector<std::vector<int> >+;
#endif

typedef std::vector<std::vector<int> > MyVV;

void nonsplit() {
   TFile* f=new TFile("nonsplit.root","RECREATE");
   TTree* t=new TTree("T","T");
   MyVV myVV;
   MyVV* pMyVV=&myVV;

   myVV.resize(100);
   for (int i=0; i<100; i++)
      myVV[i].resize(100);
   
   t->Branch("myVV",&pMyVV);
   for (int i=0; i<100; i++) {
      for (int j=0; j<100; j++)
         myVV[i][j]=(i+j)%42;
      t->Fill();
   }
   t->Write();
   delete f;
}
