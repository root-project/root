#include "TTree.h"
#include <vector>

namespace std {}; using namespace std;

#if defined(__CINT__) && !defined(__MAKECINT__)
#include "MyClass.h+"
#else
#include "MyClass.h"
#endif

#if defined(__MAKECINT__) && !defined(R__ACLIC_ROOTMAP)
#pragma link C++ class MyClass+;
#endif
#if defined(__MAKECINT__) && defined(VECTOR_DICT)
#pragma link C++ class vector<MyClass>;
#endif

void runmixing(bool scan = false) 
{
   int len = 5;

   std::vector<MyClass> *p = new std::vector<MyClass>;

   TTree *t = new TTree("tree","bad vector");
   
   t->Branch("checked_value", "vector<MyClass>", &p);
   t->Branch("value", "vector<MyClass>", (void*)&p);

   for(int i = 0; i<len; ++i) {
      p->push_back( MyClass( i*12 ) );
      if (scan) t->Fill();
   }
   t->SetBranchAddress("value",&p);

   if (scan) {
      //t->Print();
      t->Scan("*");
   }
}
