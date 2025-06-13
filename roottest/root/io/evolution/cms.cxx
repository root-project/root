#include "TFile.h"
#include "TBranch.h"
#include "TTree.h"
#include "TSystem.h"

//#define SHORT

class Obj {
public:
   virtual ~Obj(){}
};

class A {
public:
  A() : ok(1) {}
  unsigned char ok;
  // pool::Ref<Obj> d;

};


class Id {
public:
  Id(){}
  Id(int ir, int ie): r(ir), e(ie){}

  int r;
  int e;
};

class B : public A{
public:
  B(){}
  B(Id ia,Id ib) : a(ia), b(ib), s(0), cs(0), w(false){}
  B(Id ia,Id ib,
    unsigned short int is,
    unsigned short int ic) : a(ia), b(ib), s(is), cs(ic), w(false){}
  Id a;
  Id b;
#ifdef SHORT
  unsigned short int s;
  unsigned short int cs;
#else
  unsigned int s;
  unsigned int cs;
#endif
  bool w;
};

class CMSColl {
public:
  CMSColl() {}
  virtual ~CMSColl() {}
  std::vector<B> collection;
};

void CMSTestWrite()  {
  TFile* f = TFile::Open("CMS.root","RECREATE");
  printf("File version: %d\n\n",f->GetVersion());
  TTree*  t  = new TTree("test", "An example of a ROOT tree");
  CMSColl* obj = new CMSColl;
  /* TBranch* b = */ t->Branch("Coll_test", "CMSColl", &obj);
  for ( size_t i = 0; i < 10; ++i )   {
    obj->collection.push_back(B());
    t->Fill();
  }
  t->Print();
  t->Write();
}

void CMSTestRead()
{
  TFile* f = TFile::Open("CMS.root");
  printf("File version:%d\n\n",f->GetVersion());
  TTree* t=(TTree*)f->Get("test");
  CMSColl* obj = new CMSColl;
  TBranch* b = t->GetBranch("Coll_test");
  b->SetAddress(&obj);
  gDebug=0;
  printf("Read %d bytes...\n",b->GetEvent(0));
  printf("Read %d bytes...\n",b->GetEvent(1));
  printf("Read %d bytes...\n",b->GetEvent(2));
  printf("Read %d bytes...\n",b->GetEvent(3));
}

