#include <vector>
#include "TTree.h"
#include "TFile.h"
#include "Riostream.h"

#ifdef __MAKECINT__
//#pragma link C++ class pair<int,long>+;
#pragma link C++ class vector<pair<int,long> >+;
#endif

void runsimple() {
   TTree *t = new TTree("T","T");
   std::vector<int> myvec;
   myvec.push_back(3);
   t->Branch("vec",&myvec);
   t->Fill();
   delete t;
}

void write(int len = 20) {
   TFile *f = new TFile("simple.root","RECREATE");
   TTree *t = new TTree("T","T");
   std::vector<int> myvec;
   std::vector<pair<int,long> > myobj;
   
   t->Branch("vec",&myvec);
   t->Branch("obj.",&myobj);

   for(int i=0;i<len;++i) {
      if (i%10 == 0) myvec.clear();
      myvec.push_back(100+i);
      myobj.push_back(make_pair(i,100-i));
      t->Fill();
   }
   f->Write();
   delete f;
}

void read() {
   TFile *f = new TFile("simple.root");
   TTree *t; f->GetObject("T",t);
   if (!t) return;
   std::vector<int> *myvec = 0;
   std::vector<pair<int,long> > *myobj = 0;
   t->SetBranchAddress("vec",&myvec);
   t->SetBranchAddress("obj",&myobj);
   for(int i=0;i<t->GetEntries();++i) {
      t->GetEntry(i);
      cout << "single: size:  " << myvec->size() << "\t";
      cout << "alloc: " << myvec->capacity() << "\n";
      cout << "pair  : size:  " << myobj->size() << "\t";
      cout << "alloc: " << myobj->capacity() << "\n";
   }
}
      
      
  
