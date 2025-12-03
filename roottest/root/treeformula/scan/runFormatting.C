#include "TFile.h"
#include "TTree.h"
#include "TError.h"
TTree* makeTree();

void runFormatting(int mode = 0)
{
   if (mode==1 || mode ==2) {
      makeTree();
      if (mode==1) return;
   }
   TFile* file = TFile::Open("testScanIn.root");
   if(file == NULL) {
      makeTree();
      file = TFile::Open("testScanIn.root");
   }

   TTree* tree = (TTree*) file->Get("tree");
   tree->Scan("val:flag:flag:c:c:cstr", "", "col=::#x::c:");
}

TTree* makeTree()
{
  Info("testScan", "Making testScanIn.root...");
  TFile* file = TFile::Open("testScanIn.root", "RECREATE");
  TTree* tree = new TTree("tree", "tree");

  double val = 0.0;
  int flag = 0x1;
  char c = 'a';
  char *c2 = new char[4];
  c2[0] = 'i';
  c2[1] = '0';
  c2[2] = '0';
  c2[3] = 0;
  tree->Branch("val", &val, "val/D");
  tree->Branch("flag", &flag, "flag/I");
  tree->Branch("c", (void*)&c, "c/b");
  tree->Branch("cstr", (void*)c2, "cstr/C");

  for(int i=0; i<10; i++) {
    tree->Fill();
    val += 1.1;
    flag <<= 1;
    c++;
    (c2[2])++;
    printf("%f:%d:%#x:%c\n",val,flag,flag,c /* ,c2 */);
  }

  tree->Fill();
  tree->Write();
  file->Close();
  return 0;
}
