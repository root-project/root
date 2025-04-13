#include "TTree.h"
#include <vector>

class ScanStringClass {
public:
   ScanStringClass() : title("notset"),i(-1) {}
   TString title;
   int i;
};

void ScanString() {
   TTree * t = new TTree("T","T");
   ScanStringClass *s = new ScanStringClass;
   std::vector<ScanStringClass> *v = new std::vector<ScanStringClass>;
   t->Branch("top.",&s);
   t->Branch("tops.",&v);
   s->title = "three"; s->i = 3;
   v->push_back(*s);
   s->title = "two"; s->i = 2;
   v->push_back(*s);
   s->title = "one"; s->i = 1;
   v->push_back(*s);
   s->title = "alone"; s->i = 0;
   t->Fill();
   s->title = "set";
   t->Fill();
   t->Scan("tops.i:tops.title.fData","");
}

#ifdef __MAKECINT__
#pragma link C++ class vector<ScanStringClass>;
#endif

