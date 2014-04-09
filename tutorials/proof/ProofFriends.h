//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Thu May 27 19:04:42 2010 by ROOT version 5.27/03
// from TTree Tmain/Main tree for tutorial friends
// found on file: /tmp/user/master-trunk/ganis/data/0.1/pcphsft64-1274979719-4411/tree_0.1.root
//////////////////////////////////////////////////////////

#ifndef ProofFriends_h
#define ProofFriends_h

#include "TChain.h"
#include "TSelector.h"

class TH1F;
class TH2F;

class ProofFriends : public TSelector {
public :
   TTree          *fChain;   //!pointer to the analyzed TTree or TChain

   Bool_t          fPlot;    // Whether to plot the result
   Bool_t          fDoFriends; // Whether to use the friend tree

   // Specific members
   TH2F            *fXY;
   TH1F            *fZ;
   TH1F            *fR;
   TH2F            *fRZ;

   // Declaration of leaf types
   Int_t           Run;
   Long64_t        Event;
   Float_t         x;
   Float_t         y;
   Float_t         z;

   Float_t         r;     // The friend

   // List of branches
   TBranch        *b_Run;   //!
   TBranch        *b_Event;   //!
   TBranch        *b_x;   //!
   TBranch        *b_y;   //!
   TBranch        *b_z;   //!

   TBranch        *b_r;   //! The friend branch

   ProofFriends();
   virtual ~ProofFriends() { }
   virtual Int_t   Version() const { return 2; }
   virtual void    Begin(TTree *tree);
   virtual void    SlaveBegin(TTree *tree);
   virtual void    Init(TTree *tree);
   virtual Bool_t  Notify();
   virtual Bool_t  Process(Long64_t entry);
   virtual Int_t   GetEntry(Long64_t entry, Int_t getall = 0) { return fChain ? fChain->GetTree()->GetEntry(entry, getall) : 0; }
   virtual void    SetOption(const char *option) { fOption = option; }
   virtual void    SetObject(TObject *obj) { fObject = obj; }
   virtual void    SetInputList(TList *input) { fInput = input; }
   virtual TList  *GetOutputList() const { return fOutput; }
   virtual void    SlaveTerminate();
   virtual void    Terminate();

   ClassDef(ProofFriends,0);
};

#endif

#ifdef ProofFriends_cxx
void ProofFriends::Init(TTree *tree)
{
   // The Init() function is called when the selector needs to initialize
   // a new tree or chain. Typically here the branch addresses and branch
   // pointers of the tree will be set.
   // It is normally not necessary to make changes to the generated
   // code, but the routine can be extended by the user if needed.
   // Init() will be called many times when running on PROOF
   // (once per file to be processed).

   // Set branch addresses and branch pointers
   if (!tree) return;
   fChain = tree;
   fChain->SetMakeClass(1);

   fChain->SetBranchAddress("Run", &Run, &b_Run);
   fChain->SetBranchAddress("Event", &Event, &b_Event);
   fChain->SetBranchAddress("x", &x, &b_x);
   fChain->SetBranchAddress("y", &y, &b_y);
   fChain->SetBranchAddress("z", &z, &b_z);

   fChain->SetBranchAddress("r", &r, &b_r);  // The friend
}

Bool_t ProofFriends::Notify()
{
   // The Notify() function is called when a new file is opened. This
   // can be either for a new TTree in a TChain or when when a new TTree
   // is started when using PROOF. It is normally not necessary to make changes
   // to the generated code, but the routine can be extended by the
   // user if needed. The return value is currently not used.

   return kTRUE;
}

#endif // #ifdef ProofFriends_cxx
