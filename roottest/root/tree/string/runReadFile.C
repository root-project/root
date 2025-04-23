#include "TFile.h"
#include "TTree.h"

// Testing (indirectly) TLeafC::ReadValue.

void readlong() {
  TFile * fout = new TFile("longstring.root", "recreate");
  TTree * nt = new TTree("nt", "");
  nt->ReadFile("longstring.txt", "x/I:name/C");
  nt->Write();
  fout->Close();
}

void readshort() {
  TFile * fout = new TFile("shortstring.root", "recreate");
  TTree * nt = new TTree("nt", "");
  nt->ReadFile("shortstring.txt", "x/I:name/C");
  nt->Write();
  fout->Close();
}

void runReadFile() {
   readshort();
   readlong();
}
