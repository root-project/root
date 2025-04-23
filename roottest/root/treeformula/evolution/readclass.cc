// testclass.cc 
#ifdef ClingWorkAroundMissingSmartInclude
int loader = gROOT->ProcessLine(".L myclass2.cc+");
#else
#include "myclass2.cc+" 
#endif
#include "TFile.h" 
#include "TTree.h" 
#include "TPostScript.h" 
#include "TRint.h" 

void readclass() 
{ 
   TFile f("test.root"); 
   //TTree *tree = (TTree*)f.FindObjectAny("tree"); 
   TTree *tree; f.GetObject("tree",tree);
   tree->Scan("mybranch.a");
   tree->Scan("mybranch.a.a");
#ifndef ClingWorkAroundCallfuncAndInline
   tree->Scan("mybranch.GetA()");
#endif
}
