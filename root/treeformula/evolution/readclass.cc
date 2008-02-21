// testclass.cc 
#include "myclass2.cc+" 
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
   tree->Scan("mybranch.GetA()");
}
