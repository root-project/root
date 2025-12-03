// testclass.cc 
#ifdef ClingWorkAroundMissingSmartInclude
int loader = gROOT->ProcessLine(".L myclass1.cc+");
#include "myclass1.h"
#else
#include "myclass1.cc+" 
#endif
#include "TFile.h" 
#include "TTree.h" 
#include "TClonesArray.h" 


void writing() {
   TFile f("test.root","recreate"); 
   TTree *tree = new TTree("tree", "tree"); 
   TClonesArray *array = new TClonesArray("myclass"); 
   tree->Branch("mybranch", "TClonesArray", &array, 32000, 99); 
   array->ExpandCreate(1); 
   myclass * m = ((myclass*) (array->At(0)));
   m->a = 4;
   tree->Fill(); 
   array->ExpandCreate(2); 
   m = ((myclass*) (array->At(0)));
   m->a = 5;
   m = ((myclass*) (array->At(1)));
   m->a = 6;
   tree->Fill();
   f.Write(); 
   tree->Scan("a.a");
}


void reading() 
{
   TFile f("test.root"); 
   //TTree *tree = (TTree*)f.FindObjectAny("tree"); 
   TTree *tree; f.GetObject("tree",tree);
   tree->Scan("mybranch.a");
   tree->Scan("mybranch.a.a");
   tree->Scan("a.a");
}



void writeclass() 
{ 
   writing();
   reading();
}
