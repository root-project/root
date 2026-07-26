#include <vector>
#include "TTree.h"

#ifdef __ROOTCLING__
#pragma link C++ class A;
#pragma link C++ class B+;
#pragma link C++ class std::vector<B>+;
#endif

struct A {
    int x;
    ClassDef(A, 1);
};

struct B : A {
    int y;
    ClassDef(B, 1);
};

void oldNewIOMix() {
   TTree* tree = new TTree("T", "T");
   std::vector<B> bvec;
   tree->Branch("B", &bvec);
}
