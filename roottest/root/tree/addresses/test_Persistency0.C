#include <string>
#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "TSystem.h"
#include "TROOT.h"

#include <cmath>
#include <iostream>

//--Persistency tests
//
struct Data {
  int    i;
  long   l;
  char   c;
  short  s;
  float  f;
  double d;
  bool   b;
  Data() : i(0), l(0), c(0), s(0), f(0.0), d(0.0), b(false) {
    //std::cout << "Data constructor called" << std::endl; 
  }
  Data(const Data& o) : i(o.i), l(o.l), c(o.c), s(o.s), f(o.f), d(o.d), b(o.b) {
    //std::cout << "Data copy constructor called" << std::endl; 
  }
  ~Data() {
    //std::cout << "Data destructor called" << std::endl; 
  }
  void setPattern(int o = 0) {
    c = 9 + o;
    s = 999 + o;
    i = 9999 + o;
    l = 99999 + o;
    b = o % 2 == 0;
    f = 9.9F + o;
    d = 9.99 + o;
  }  
  bool checkPattern(int o = 0) {
    return (c == 9 + o)  &&
           (s == 999 + o) &&
           (i == 9999 + o) &&
           (l == 99999 + o) &&
           (b == (o % 2 == 0)) &&
           (std::fabs(f - (9.9F + o)) < 0.0001F ) &&
           (std::fabs(d - (9.99 + o)) < 0.000001 );
  }
};

struct OData {
  int   i2;
  float f2;
  char  c2;
  short s2;
  OData() : i2(0), f2(0), c2(0), s2(0) {}
};


struct Aggregate : public Data {
  int  extra;
  Data  dval1;
  Data  dval2;
  Data* dptr;
  Aggregate() : extra(0), dptr(0) {}
  ~Aggregate() { delete dptr; }
};

struct Final : public OData, public Aggregate {
  Data dval;
};


template <class T1, class T2> 
void failUnlessEqual( T1 r, T2 e, const char* c = "") { 
  if( r != e ) {
  cout << "Test failed in " << c << " : got " << r << " expected " << e << endl;
  assert(false);
  }
}
template <class T1> 
void failUnless( T1 r, const char* c = "") { 
   if( ! r ) {
      cout << "Test failed in " << c << " : got " << r << endl;
      assert(false);
   }
}


bool test_ObjectInitialization() {
  Final f;
  f.setPattern(10);   // base class Data from Aggregate 
  f.dval.setPattern(0);
  f.dval1.setPattern(1);
  f.dval2.setPattern(2);
  f.dptr = new Data();
  f.dptr->setPattern(3);
  failUnless(f.checkPattern(10),"Direct base class");
  failUnless(f.dval.checkPattern(0), "Direct aggregation");
  failUnless(f.dval1.checkPattern(1),"Aggregation through a base class 1");
  failUnless(f.dval2.checkPattern(2),"Aggregation through a base class 2");
  failUnless(f.dptr->checkPattern(3),"By pointer");
  return true;
}

bool test_WriteObject() {

  Final f0;
  Final f1;
  f1.setPattern(10); 
  f1.dval.setPattern(0);
  f1.dval1.setPattern(1);
  f1.dval2.setPattern(2);
  f1.dptr = new Data();
  f1.dptr->setPattern(3);

  Data d0;
  Data d1;
  d1.setPattern(1);

  TFile fo("data.root","RECREATE");
  failUnless(fo.WriteObject(&f0,"f0"),"writing f0");
  failUnless(fo.WriteObject(&f1,"f1"),"writing f1");
  failUnless(fo.WriteObjectAny(&d0,"Data","d0"),"writing d0");
  failUnless(fo.WriteObjectAny(&d1,"Data","d1"),"writing d1");

  fo.Close();
  return true;
}
bool test_ReadObject() {

  TFile fi("data.root");
  Final* f;
  fi.GetObject("f1", f);
  failUnless(f->checkPattern(10),"Indirect base class");
  failUnless(f->dval.checkPattern(0), "Direct aggregation");
  failUnless(f->dval1.checkPattern(1),"Aggregation through a base class 1");
  failUnless(f->dval2.checkPattern(2),"Aggregation through a base class 2");
  failUnless(f->dptr->checkPattern(3),"By pointer");

  Data* d = (Data*)fi.FindObjectAny("d1");
  failUnless(d->checkPattern(1),"Direct");

  fi.Close();
  return true;
}
bool test_WriteObjectInTree(bool withdot) {
  Data* d      = new Data();
  Aggregate* g = new Aggregate();
  Final* f     = new Final();

  TFile fo("tree.root","RECREATE");
  TTree* tree = new TTree("tree","Test Tree");
  if (withdot) {
     failUnless( tree->Branch("Data.","Data",&d, 32000,99), "Creating Data branch");
     failUnless( tree->Branch("Aggregate.","Aggregate",&g, 32000,99), "Creating Aggregate branch");
     failUnless( tree->Branch("Final.","Final",&f, 32000,99), "Creating Final branch");
  } else {
     failUnless( tree->Branch("Data","Data",&d, 32000,99), "Creating Data branch");
     failUnless( tree->Branch("Aggregate","Aggregate",&g, 32000,99), "Creating Aggregate branch");
     failUnless( tree->Branch("Final","Final",&f, 32000,99), "Creating Final branch");
  }
  //tree->Print();
  for (int i = 0; i < 2; i++ ) {
    d->setPattern(i);

    g->setPattern(i); 
    g->dval1.setPattern(i+10);
    g->dval2.setPattern(i+20);
    g->dptr = new Data();
    g->dptr->setPattern(i+30);

    f->setPattern(i+10); 
    f->dval.setPattern(i+20);
    f->dval1.setPattern(i+30);
    f->dval2.setPattern(i+40);
    f->dptr = new Data();
    f->dptr->setPattern(i+50);
    //d->IsA()->Dump(&f->dval1);

    failUnless( tree->Fill(), "Filling tree");
  }
  fo.Write();
  fo.Close();
  return true;
}

bool test_ReadObjectInTree(bool withdot) {
  Data* d      = new Data();
  Aggregate* g = new Aggregate();
  Final* f     = new Final(); 

  TFile fi("tree.root");
  TTree* tree = (TTree*)fi.Get("tree");
  failUnless( tree, "Getting tree");
  if (withdot) {
     tree->GetBranch("Data.")->SetAddress(&d);
     tree->GetBranch("Aggregate.")->SetAddress(&g);
     tree->GetBranch("Final.")->SetAddress(&f);
  } else {
     tree->GetBranch("Data")->SetAddress(&d);
     tree->GetBranch("Aggregate")->SetAddress(&g);
     tree->GetBranch("Final")->SetAddress(&f);
  }
  int n = tree->GetEntries();
  failUnlessEqual( 2, n, "Number of entries in Tree");
  for ( int i = 0; i < n; i++ ) {
    failUnless( tree->GetEntry(i), "Getting Entry");
    failUnless( d->checkPattern(i), "Data pattern");
    failUnless( g->checkPattern(i), "Aggregate");
    failUnless( g->dval1.checkPattern(i+10), "Aggregate dval1");
    failUnless( g->dval2.checkPattern(i+20), "Aggregate dvla2");
    failUnless( g->dptr->checkPattern(i+30), "Aggregate dptr");
    failUnless( f->checkPattern(i+10), "In Aggregate base");
    failUnless( f->dval.checkPattern(i+20), "Final dval");
    //gROOT->GetClass("Data")->Dump(&f->dval1);
    failUnless( f->dval1.checkPattern(i+30), "Final dval1");
    failUnless( f->dval2.checkPattern(i+40), "Final dval2");
    failUnless( f->dptr->checkPattern(i+50), "Final dptr");
  }
  fi.Close();
  return true;
}
void test_Persistency0(bool withdot=false)
{
  gROOT->ProcessLine(".O 0");  // Disable CINT optimization

  cout << "ObjectInitialization: "      << (test_ObjectInitialization()      ? "OK" : "FAIL") << endl;
  cout << "WriteObject:          "      << (test_WriteObject()       ? "OK" : "FAIL") << endl;
  cout << "ReadObject:           "      << (test_ReadObject()        ? "OK" : "FAIL") << endl;
  cout << "WriteObjectInTree:    "      << (test_WriteObjectInTree(withdot) ? "OK" : "FAIL") << endl;
  cout << "ReadObjectInTree:     "      << (test_ReadObjectInTree(withdot) ? "OK" : "FAIL") << endl;
}
