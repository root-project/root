// -*- mode: c++ -*- 
//////////////////////////////////////////////////////////
//   This class has been automatically generated 
//     (Fri Mar 22 19:21:08 2002 by ROOT version3.03/01)
//   from TTree tree/tree
//   found on file: file.root
//////////////////////////////////////////////////////////


#ifndef tree_h
#define tree_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
const Int_t kMaxfoo = 3;

class tree {
public :
  TTree          *fChain;   //!pointer to the analyzed TTree or TChain
  Int_t           fCurrent; //!current Tree number in a TChain

   //Declaration of leaves types
  Int_t           foo_;
  UInt_t          foo_fUniqueID[kMaxfoo];//[foo_]
  UInt_t          foo_fBits[kMaxfoo];    //[foo_]
  Int_t           foo_fFoo[kMaxfoo];     //[foo_]

   //List of branches
  TBranch        *b_foo_;                //!
  TBranch        *b_foo_fUniqueID;       //!
  TBranch        *b_foo_fBits;           //!
  TBranch        *b_foo_fFoo;            //!

  tree(TTree *tree=0);
  ~tree();
  Int_t  Cut(Int_t entry);
  Int_t  GetEntry(Int_t entry);
  Int_t  LoadTree(Int_t entry);
  void   Init(TTree *tree);
  void   Loop();
  Bool_t Notify();
  void   Show(Int_t entry = -1);
};

#endif

#ifdef tree_cxx
tree::tree(TTree *tree)
{
  // if parameter tree is not specified (or zero), connect the file
  // used to generate this class and read the Tree.
  if (tree == 0) {
    TFile *f = (TFile*)gROOT->GetListOfFiles()->FindObject("file.root");
    if (!f) 
      f = new TFile("file.root");

    tree = (TTree*)gDirectory->Get("tree");
    
  }
  Init(tree);
}

tree::~tree()
{
  if (!fChain) 
    return;
  delete fChain->GetCurrentFile();
}

Int_t tree::GetEntry(Int_t entry)
{
  // Read contents of entry.
  if (!fChain) 
    return 0;
  return fChain->GetEntry(entry);
}

Int_t tree::LoadTree(Int_t entry)
{
  // Set the environment to read one entry
  if (!fChain) 
    return -5;
  Int_t centry = fChain->LoadTree(entry);
  if (centry < 0) 
    return centry;
  if (fChain->IsA() != TChain::Class()) 
    return centry;
  TChain *chain = (TChain*)fChain;
  if (chain->GetTreeNumber() != fCurrent) {
    fCurrent = chain->GetTreeNumber();
    Notify();
  }
  return centry;
}

void tree::Init(TTree *tree)
{
  //   Set branch addresses
  if (tree == 0) 
    return;
  fChain    = tree;
  fCurrent = -1;
  fChain->SetMakeClass(1);
  
  fChain->SetBranchAddress("foo",&foo_);
  fChain->SetBranchAddress("foo.fUniqueID",foo_fUniqueID);
  fChain->SetBranchAddress("foo.fBits",foo_fBits);
  fChain->SetBranchAddress("foo.fFoo",foo_fFoo);
  Notify();
}

Bool_t tree::Notify()
{
  // Called when loading a new file.
  // Get branch pointers.
  b_foo_          = fChain->GetBranch("foo");
  b_foo_fUniqueID = fChain->GetBranch("foo.fUniqueID");
  b_foo_fBits     = fChain->GetBranch("foo.fBits");
  b_foo_fFoo      = fChain->GetBranch("foo.fFoo");
  return kTRUE;
}

void tree::Show(Int_t entry)
{
  // Print contents of entry.
  // If entry is not specified, print current entry
  if (!fChain) 
    return;
  fChain->Show(entry);
}
Int_t tree::Cut(Int_t entry)
{
  // This function may be called from Loop.
  // returns  1 if entry is accepted.
  // returns -1 otherwise.
  return 1;
}
#endif // #ifdef tree_cxx

