#include "TVector3.h"
#include "TFile.h"
#include "TTree.h"

void execBranchObjSplit0() {
  TVector3 Pos(1, 2, 3);
  TVector3 *ptr0 = new TVector3(4,5,6);
  TFile *f = TFile::Open("test.root", "recreate");
  TTree *t = new TTree("tree", "tree with vector");
  t->Branch("Pos", &Pos, 32000, 0); // segmentation violation if splitlevel = 0
  //fprintf(stderr,"autodelete = %d\n",t->GetBranch("Pos")->IsAutoDelete());
  t->Fill();
  t->Write();
  t->Scan("Pos.X()");
  delete f;
  f = TFile::Open("test.root");
  f->GetObject("tree",t);
  TVector3 *ptr = 0; // new TVector3;
  t->SetBranchAddress("Pos",&ptr);
  //fprintf(stderr,"autodelete = %d\n",t->GetBranch("Pos")->IsAutoDelete());
  t->GetEntry(0);
  // ptr->Dump();
  t->Scan("Pos.fX");
  delete f;
}
