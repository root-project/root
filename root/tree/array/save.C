#include "TFile.h"
#include "TSystem.h"
#include "TestObj.h"
#include "TTree.h"

void save() 
{

//gROOT->ProcessLine(".L TestObj.cpp+");
//gSystem->Load("TestObj.so");

AllReconstructions* pRec = new AllReconstructions;
TFile *file = new TFile("test.root","RECREATE");
TTree *Tree = new TTree("TestTree","Test"); 
Tree->Branch("Main", "AllReconstructions", &pRec,32000,99);
pRec->Init();
for (int i = 0; i < 10; ++i) { 
   pRec->Init(i); 
   Tree->Fill(); 
};
file->Write();
pRec->print();

delete file;

pRec=0;

fprintf(stderr,"reloading\n");
file = new TFile("test.root");
TTree *tree = (TTree*)file->Get("TestTree");
tree->SetBranchAddress("Main",&pRec);
tree->GetEntry(3);
pRec->print();
}
