R__LOAD_LIBRARY(stringarray_h)
#include "stringarray.h"
#include "TFile.h"
#include "TTree.h"

void execreadfile ()
{
   auto f = new TFile("stringarray.root", "READ");
   auto t = f->Get<TTree>("t");
   std::vector<mystrarray> *vecptr = nullptr;
   container *cont = nullptr;
   t->SetBranchAddress("vec", &vecptr);
   t->SetBranchAddress("cont.", &cont);
   t->GetEntry(0);
}

