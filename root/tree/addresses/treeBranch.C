#include "TTree.h"
#include "TROOT.h"

#if defined(__CINT__) && !defined(__MAKECINT__)
// do nothing
#else
#include "userClass.C"
#endif

void treeBranch() 
{
   if (TClass::GetClass("TopLevel")==0) gROOT->ProcessLine(".L userClass.C+");
   TopLevel *one = new BottomOne;
#if !defined(__CINT__) && !defined(__ROOTCLING__)
   TopLevel *missing = new BottomMissing;
#endif

   TTree * t = new TTree;
   t->Branch("one",&one);
#if !defined(__CINT__) && !defined(__ROOTCLING__)
   t->Branch("missing",&missing);
#endif
   t->Print();
}
