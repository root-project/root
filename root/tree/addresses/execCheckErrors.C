#include "TFile.h"
#include "TChain.h"
#include "TBranch.h"
#include "TString.h"
#include "TObjString.h"

#include <iostream>

void execCheckErrors(bool what = 0)
{
  TTree *tree = 0;
  TFile *file = 0;
  if (what == 0) {
    tree = new TTree("ntuple","");
    float px = 0;
    tree->Branch("px",&px);
    TString str;
    tree->Branch("str",&str);
    tree->Fill();
    tree->ResetBranchAddresses(); 
  } else if (what == 1) {
    file = new TFile("hsimple.root");
    tree = (TTree*)file->Get("ntuple");
  } else {
    tree = new TChain("ntuple", "ntuple");
    ((TChain*)tree)->Add("hsimple.root");
  }
  // tree->SetBranchStatus("*", 1);
  // tree->SetBranchStatus("px", 1);
  // std::cout << "fTreeNumber = " << tree->GetTreeNumber() << std::endl;
  tree->LoadTree(0); // REQUIRED
  // std::cout << "fTreeNumber = " << tree->GetTreeNumber() << std::endl;
  
  TBranch *p = ((TBranch *)-1);
  Int_t r;
  
  Float_t px, fake_px;
  Float_t *pxp = new Float_t();
  Float_t *fake_pxp = new Float_t();
  
  Int_t ix;
  Int_t *ixp = new Int_t();
  
  TString s;
  TString *sp = new TString();
  
  TObjString os;
  TObjString *osp = new TObjString();
  
  // Float_t ... (should be fine)
  std::cout << std::endl << "ALL should be FINE ... " << std::endl;
  p = ((TBranch *)-1);
  r = tree->SetBranchAddress("px", &px, &p);
  std::cout << "Float_t ... " << r << std::endl;
  if (p==((TBranch*)-1)) {
     std::cout << "p unchanged\n";
  } else if (p==0) {
     std::cout << "p set to zero\n";
  } else {
     std::cout << "p set to the branch address\n";
  }
  r = tree->SetBranchAddress("px", &px);
  std::cout << "Float_t ... "  << r << std::endl;
  r = tree->SetBranchAddress("px", pxp);
  std::cout << "Float_t ... "  << r << std::endl;
  r = tree->SetBranchAddress("px", &pxp);
  std::cout << "Float_t ... "  << r << std::endl;
  
  // Int_t ...  (should fail)
  std::cout << std::endl << "ALL should FAIL ... " << std::endl;
  p = ((TBranch *)-1);
  r = tree->SetBranchAddress("px", &ix, &p);
  std::cout << "Int_t ...  "  << r << std::endl;
  if (p==((TBranch*)-1)) {
     std::cout << "p unchanged\n";
  } else if (p==0) {
     std::cout << "p set to zero\n";
  } else {
     std::cout << "p set to the branch address\n";
  }
  r = tree->SetBranchAddress("px", &ix);
  std::cout << "Int_t ...  "  << r << std::endl;
  r = tree->SetBranchAddress("px", ixp);
  std::cout << "Int_t ...  "  << r << std::endl;
  r = tree->SetBranchAddress("px", &ixp);
  std::cout << "Int_t ...  "  << r << std::endl;
  
  // TString ... (should fail)
  std::cout << std::endl << "ALL should FAIL ... " << std::endl;
  p = ((TBranch *)-1);
  r = tree->SetBranchAddress("px", &s, &p);
  std::cout << "TString ... "  << r << std::endl;
  if (p==((TBranch *)-1)) {
     std::cout << "p unchanged\n";
  } else if (p==0) {
     std::cout << "p set to zero\n";
  } else {
     std::cout << "p set to the branch address\n";
  }
  r = tree->SetBranchAddress("px", sp);
  std::cout << "TString ... "  << r << std::endl;
  r = tree->SetBranchAddress("px", &sp);
  std::cout << "TString ... "  << r << std::endl;
  
  // TObjString ... (should fail)
  std::cout << std::endl << "ALL should FAIL ... " << std::endl;
  p = ((TBranch *)-1);
  r = tree->SetBranchAddress("px", &os, &p);
  std::cout << "TObjString ... "  << r << std::endl;
  if (p==((TBranch*)-1)) {
     std::cout << "p unchanged\n";
  } else if (p==0) {
     std::cout << "p set to zero\n";
  } else {
     std::cout << "p set to the branch address\n";
  }
  r = tree->SetBranchAddress("px", osp);
  std::cout << "TObjString ... "  << r << std::endl;
  r = tree->SetBranchAddress("px", &osp);
  std::cout << "TObjString ... "  << r << std::endl;
  
  // nonexistent branch ... (should fail)
  std::cout << std::endl << "ALL should FAIL ... " << std::endl;
  p = ((TBranch *)-1);
  r = tree->SetBranchAddress("fake_px", &fake_px, &p);
  std::cout << "nonexistent branch ... "  << r << std::endl;
  if (p==((TBranch*)-1)) {
     std::cout << "p unchanged\n";
  } else if (p==0) {
     std::cout << "p set to zero\n";
  } else {
     std::cout << "p set to the branch address\n";
  }
  r = tree->SetBranchAddress("fake_px", &fake_px);
  std::cout << "nonexistent branch ... "  << r << std::endl;
  r = tree->SetBranchAddress("fake_px", fake_pxp);
  std::cout << "nonexistent branch ... "  << r << std::endl;
  r = tree->SetBranchAddress("fake_px", &fake_pxp);
  std::cout << "nonexistent branch ... "  << r << std::endl;

  // Float for TString ... (should fail)
  std::cout << std::endl << "ALL should FAIL ... " << std::endl;
  p = ((TBranch *)-1);
  r = tree->SetBranchAddress("str", &px, &p);
  std::cout << "Float ... "  << r << std::endl;
  if (p==((TBranch*)-1)) {
     std::cout << "p unchanged\n";
  } else if (p==0) {
     std::cout << "p set to zero\n";
  } else {
     std::cout << "p set to the branch address\n";
  }
  r = tree->SetBranchAddress("str", &px);
  std::cout << "Float ... "  << r << std::endl;
  r = tree->SetBranchAddress("str", pxp);
  std::cout << "Float ... "  << r << std::endl;
  r = tree->SetBranchAddress("str", &pxp);
  std::cout << "Float ... "  << r << std::endl;
  
   
  TChain *ch = new TChain("MonoData");
  ch->Add("memleak.root");
  ch->LoadTree(0);
  r = ch->SetBranchAddress("mono", &sp, &p);
  std::cout << "From chain, TString ... "  << r << std::endl;

  TTree *treech = ch;
  r = treech->SetBranchAddress("mono", &sp, &p);
  std::cout << "From tree, TString ... "  << r << std::endl;
   
}
