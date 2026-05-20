#include "TTree.h"
#include "TROOT.h"

#if defined(__CLING__) && !defined(__ROOTCLING__)
// do nothing
#else
#include "userClass.C"
#endif

void treeBranch() 
{
   if (TClass::GetClass("TopLevel")==0) gROOT->ProcessLine(".L userClass.C+");
   TopLevel *one = new BottomOne;
#if !defined(__CLING__)
   TopLevel *missing = new BottomMissing;
#endif

   TTree * t = new TTree;
   t->Branch("one",&one);
#if !defined(__CLING__)
   t->Branch("missing",&missing);
#endif
   t->Print();
}
