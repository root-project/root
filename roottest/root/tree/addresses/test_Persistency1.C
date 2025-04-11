#include "TROOT.h"
#include "TTree.h"
#include "TBranch.h"
#include <iostream>
#include "TFile.h"
#include <vector>

static int s_smartRefCount = 0;

struct ITime  {
  virtual ~ITime() {}
  virtual Long64_t getData() = 0;
};
struct TimePoint : public ITime  {
  Long64_t m_time;
  TimePoint() : m_time(0) {}
  ~TimePoint() override {}
  Long64_t getData() override { return m_time; }
};
struct SmartRefBase  {
  int m_linkID,m_hintID;
  SmartRefBase() : m_linkID(0), m_hintID(0) {
    s_smartRefCount++;
  }
  SmartRefBase(const SmartRefBase& b) : m_linkID(b.m_linkID), m_hintID(b.m_hintID) {
    s_smartRefCount++;
  }
  ~SmartRefBase() { 
    s_smartRefCount--; 
    if ( s_smartRefCount < 0 )  {
      std::cout << "Severe error: Multiple non-allowed object deletion!!!!" << std::endl;
      exit(1);
    }
  }
  SmartRefBase& operator=(const SmartRefBase& b)  { 
    m_linkID = b.m_linkID; 
    m_hintID = b.m_hintID; 
    return *this;
  }
};
template <class T> struct SmartRef   {
  SmartRefBase m_base;
  SmartRef(const SmartRef& c) : m_base(c.m_base)  {     }
  SmartRef()  {     }
  ~SmartRef() {     }
  SmartRef& operator=(const SmartRef& b)  {
    m_base = b.m_base;
    return *this; 
  }
};
template <class T> struct SmartRefVector : public std::vector<SmartRef<T> > {
  SmartRefVector() {}
  ~SmartRefVector() {}
};

struct DataObject  {
  int f; //! transient
  DataObject() : f(0) {}
  virtual ~DataObject() {}
};

struct Obj {
  Obj() {
  }
  virtual ~Obj() {
  }
};

struct Event : public DataObject  {
  int m_evt;
  int m_run;
  TimePoint m_time;
  SmartRefVector<Obj> m_collisions;
  Event() : m_evt(0) , m_run(0) {}
  ~Event() override { m_collisions.clear(); }
};

void writeTree()  {
  SmartRef<Obj> ref;
  Event* d = new Event();
  TFile f("dummy1.root","RECREATE");
  TTree* t = new TTree("T","T");
  /* TBranch* b = */ t->Branch("Event","Event",&d);
  std::cout << "Filling: ";
  for ( int i=0; i < 10; ++i )  {
    d->m_evt = d->m_run = i;
    for ( int j=0; j<10; ++j )  {
      ref.m_base.m_linkID = i;
      ref.m_base.m_hintID = j;
      d->m_collisions.push_back(ref);
    }
    std::cout << d->m_collisions.size() << " .. ";
    t->Fill();
  }
  std::cout << std::endl;
  delete d;
  f.Write();
  f.Close();
}

void readTree()  {
  std::cout << "Reading" << std::endl;
  Event* d = new Event();
  TFile f("dummy1.root");
  f.ls();
  TTree* t = (TTree*)f.Get("T");
  t->GetBranch("Event")->SetAddress(&d);
  int n = t->GetEntries();
  //gROOT->GetClass("Event")->GetStreamerInfo()->ls();
  //gROOT->GetClass("TimePoint")->GetStreamerInfo()->ls();
  std::cout << "Reading " << n << " entries" << std::endl;
  for ( int i=0; i < n; ++i )  {
    std::cout << "Getting " << t->GetEntry(i);
    std::cout << " Event:" << d->m_evt << " " << d->m_collisions.size() << " " 
              << d->m_collisions[10*i+9].m_base.m_linkID << " " << d->m_collisions[10*i+9].m_base.m_hintID << std::endl;
  }
  delete d;
  f.Close();
}


bool test_Persistency1() {
  writeTree();
  printf("**** SmartRefCount=%d\n",s_smartRefCount);
  readTree();
  printf("**** SmartRefCount=%d\n",s_smartRefCount);
  if (s_smartRefCount!=0) return true;
  else return false;
}
