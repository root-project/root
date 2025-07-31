#include "TTree.h"
#include "TFile.h"

#include <vector>
#include <iostream>

#ifdef __MAKECINT__
//#pragma link C++ class pair<int,long>+;
#pragma link C++ class vector<pair<int,long> >+;
#endif

void simple()
{
   TTree *t = new TTree("T","T");
   std::vector<int> myvec;
   myvec.push_back(3);
   t->Branch("vec",&myvec);
   t->Fill();
   delete t;
}

void write(int len = 20)
{
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

void read()
{
   TFile *f = new TFile("simple.root");
   TTree *t; f->GetObject("T",t);
   if (!t) return;
   std::vector<int> *myvec = nullptr;
   std::vector<pair<int,long> > *myobj = nullptr;
   t->SetBranchAddress("vec",&myvec);
   t->SetBranchAddress("obj",&myobj);
   for(int i = 0; i < t->GetEntries(); ++i) {
      t->GetEntry(i);
      std::cout << "single: size:  " << myvec->size() << std::endl;
      // capacity may differ on different platforms - so do not print it
      // cout << "alloc: " << myvec->capacity() << "\n";
      std::cout << "pair  : size:  " << myobj->size() << std::endl;
      // capacity may differ on different platforms - so do not print it
      // cout << "alloc: " << myobj->capacity() << "\n";
   }
}

void runsimple()
{
   simple();
   write();
   read();
}
