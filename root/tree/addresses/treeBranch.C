#include "TTree.h"

#if defined(__CINT__) && !defined(__MAKECINT__)
// do nothing
#else
#include "userClass.C"
#endif

void treeBranch() 
{
   if (gROOT->GetClass("TopLevel")==0) gROOT->ProcessLine(".L userClass.C+");
   TopLevel *one = new BottomOne;
#ifndef __CINT__
   TopLevel *missing = new BottomMissing;
#endif

   TTree * t = new TTree;
   t->Branch("one",&one);
#ifndef __CINT__
   t->Branch("missing",&missing);
#endif
   t->Print();
}