#ifdef __CINT__


#include <vector>

#pragma link C++ class std::vector<std::pair<Char_t, UChar_t> >+;
#pragma link C++ class std::pair<Char_t, UChar_t>+;

#endif

#include <Rtypeinfo.h>
#include "Riostream.h"
#include "TVirtualCollectionProxy.h"
#include "TEmulatedCollectionProxy.h"

void whatis(TVirtualCollectionProxy* p) {
   if (p) cout << typeid(*p).name() << endl;
}

void check(TClass *c) {
   if (c==0) {
      cerr << "Missing class in check\n";
   }
   TVirtualCollectionProxy *p = c->GetCollectionProxy();
   string bad = typeid(TEmulatedCollectionProxy).name();
   string what = typeid(*p).name();
   if (bad==what) {
      cerr << "We have an emulated proxy for class " << c->GetName() << endl;
   }
}

template <class T> class tp {
public:
   tp() : value(0) {}
   T value;
};

class regular {
   Int_t val1;
   tp<Int_t> val2;
   vector<Int_t> val3;
   std::vector<std::pair<Char_t, UChar_t> > val4;
public:
   regular() : val1(0) {}
   int get() { return val1*sizeof(val2)*val3.size()*val4.size(); }
};

#ifdef __MAKECINT__
#pragma link C++ class std::vector<std::pair<Char_t, UChar_t> >+;
#pragma link C++ class std::pair<Char_t, UChar_t>+;
#pragma link C++ class regular+;
#pragma link C++ class tp<Int_t>+;
#endif

#include <TFile.h>

void write2file(const char*filename="pairs.root",int debug =0) {
   TFile *f = new TFile(filename,"RECREATE");
   regular r;
   f->WriteObject(&r,"myr");
   f->Write();
   if (debug) cout << "wrote " << filename << endl;
}

void readfile(const char*filename="pairs.root",int debug = 0) {
   TFile *f = new TFile(filename);
   if (debug) {
      whatis(TClass::GetClass("vector<Int_t>")->GetCollectionProxy());
      whatis(TClass::GetClass("vector<std::pair<Char_t, UChar_t> >")->GetCollectionProxy());
      cout << "tp<Int_t> " << (void*) TClass::GetClass("tp<Int_t>") << endl;
   }
   check(TClass::GetClass("vector<Int_t>"));
   check(TClass::GetClass("vector<std::pair<Char_t, UChar_t> >"));
   regular *r;
   f->GetObject("myr",r);
}
 
void runpairs(int what=3, const char*filename="pairs.root",int debug = 0) {
   if (what&1) write2file(filename,debug);
   if (what&2) readfile(filename,debug);
}
