#include "TROOT.h"
#include "TTree.h"
#include "TBranch.h"
#include "TFile.h"
#include <iostream>

class OOOId  {
public:
  long id;  
  static long next()     { static long cnt=0; return ++cnt; }
  OOOId():id(0) {      /* id = next(); */  }
  OOOId(long anId): id(anId) {  }
  OOOId(const OOOId& aId): id(aId.id)  {}
  virtual ~OOOId(){}
};

class TestClass  {
public:
   long a;
   OOOId m_id;  

   TestClass(int seed):      a(seed),m_id(seed){     }
   TestClass():      a(0),m_id(0){     }
    virtual ~TestClass() {    }
    int id() const {      return m_id.id;    }
    bool operator==(const TestClass& t) const {
      bool ret = true;
      if(a!=t.a) ret = false;
      if(m_id.id!=t.m_id.id) ret = false;
      return ret;
    }
    void dump() const {
      std::cout << "a="<<a<<std::endl;
      std::cout << "id="<<m_id.id<<std::endl;
    }
};

class DerivedClass : public TestClass  {
 public:
  DerivedClass(int i) : TestClass(i) {}
  DerivedClass() :TestClass() { }
  ~DerivedClass() override {}
};

void writeTree(bool withdot)  {
  DerivedClass* d = new DerivedClass(0);
  TFile f("dummy3.root","RECREATE");
  TTree* t = new TTree("T","T");
  if (withdot) t->Branch("DerivedClass.",&d);
  else t->Branch("DerivedClass",&d);
  std::cout << "Filling: ";
  for ( int i=0; i < 10; ++i )  {
    d->a = i;
    d->m_id.id = 10*i;
    t->Fill();
  }
  std::cout << std::endl;
  delete d;
  f.Write();
  f.Close();
}

void readTree(bool withdot)  {
  std::cout << "Reading" << std::endl;
  DerivedClass* d = new DerivedClass(0);
  TFile f("dummy3.root");
  //f.ls();
  TTree* t = (TTree*)f.Get("T");
  //t->Print();
  //t->GetBranch("DerivedClass.")->SetAddress(&d);
  if (withdot) t->SetBranchAddress("DerivedClass.",&d);
  else t->SetBranchAddress("DerivedClass",&d);
  int n = t->GetEntries();
  //gROOT->GetClass("DerivedClass")->GetStreamerInfo()->ls();
  //gROOT->GetClass("TestClass")->GetStreamerInfo()->ls();
  std::cout << "Reading " << n << " entries" << std::endl;
  for ( int i=0; i < n; ++i )  {
    std::cout << "Getting " << t->GetEntry(i);
    std::cout << " DerivedClass:" << d->a << " " << d->m_id.id << " " << std::endl;
  }
}


void test_Persistency3(bool withdot=false) {
  writeTree(withdot);
  printf("**** Write finished...\n");
  readTree(withdot);
  printf("**** Read finished...\n");
}
