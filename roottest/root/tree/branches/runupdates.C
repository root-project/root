#include "TFile.h"
#include "TTree.h"
#include "Riostream.h"

struct mys {
   Float_t x[3];
};
mys mys_t;

void runupdates() { 

   TFile *f = new TFile("blah.root","RECREATE"); 
   f->cd(); 

   mys_t.x[0] = 1.1; 
   mys_t.x[1] = 12.2; 
   mys_t.x[2] = 13.3; 

   TTree *t = new TTree("mytree","mytree"); 
   t->Branch("mybranch",&mys_t,"x[3]/F"); 
   t->Fill(); 
   t->Write(); 
   f->Close(); 

   for (int i = 0; i < 500; i++) { 
      std::cerr << " i = " << i << std::endl; 
      TFile *fnew = new TFile("blah.root","UPDATE"); 
      TTree *tnew = (TTree*)fnew->Get("mytree");
      tnew->SetBranchAddress("mybranch",&mys_t);
      if (i % 3 == 0) tnew->Fill(); 
      tnew->Write("",TObject::kOverwrite); 
      delete fnew; 
   } 
}
