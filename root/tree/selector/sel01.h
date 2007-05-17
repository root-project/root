//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Tue May 15 16:36:32 2007 by ROOT version 5.15/07
// from TTree T/An example of a ROOT tree
// found on file: Event.root
//////////////////////////////////////////////////////////

#ifndef sel01_h
#define sel01_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TSelector.h>
#include <TRef.h>
#include <TBits.h>

class sel01 : public TSelector {
public :
   TTree          *fChain;   //!pointer to the analyzed TTree or TChain

   // Declaration of leave types
 //Event           *event;
   Char_t          fType[20];
   Int_t           fNtrack;
   Int_t           fNseg;
   Int_t           fNvertex;
   UInt_t          fFlag;
 //EventHeader     fEvtHdr;
   TRef            fLastTrack;
   TRef            fWebHistogram;

   // List of branches
   TBranch        *b_event_fType;   //!
   TBranch        *b_event_fNtrack;   //!
   TBranch        *b_event_fNseg;   //!
   TBranch        *b_event_fNvertex;   //!
   TBranch        *b_event_fFlag;   //!
   TBranch        *b_event_fLastTrack;   //!
   TBranch        *b_event_fWebHistogram;   //!

   sel01(TTree * /*tree*/ =0) { }
   virtual ~sel01() { }
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

   ClassDef(sel01,0);
};

#endif

#ifdef sel01_cxx
void sel01::Init(TTree *tree)
{
   // The Init() function is called when the selector needs to initialize
   // a new tree or chain. Typically here the branch addresses and branch
   // pointers of the tree will be set.
   // It is normaly not necessary to make changes to the generated
   // code, but the routine can be extended by the user if needed.
   // Init() will be called many times when running on PROOF
   // (once per file to be processed).

   // Set branch addresses and branch pointers
   if (!tree) return;
   fChain = tree;
   fChain->SetMakeClass(1);

   fChain->SetBranchAddress("fType[20]", fType, &b_event_fType);
   fChain->SetBranchAddress("fNtrack", &fNtrack, &b_event_fNtrack);
   fChain->SetBranchAddress("fNseg", &fNseg, &b_event_fNseg);
   fChain->SetBranchAddress("fNvertex", &fNvertex, &b_event_fNvertex);
   fChain->SetBranchAddress("fFlag", &fFlag, &b_event_fFlag);
   fChain->SetBranchAddress("fLastTrack", &fLastTrack, &b_event_fLastTrack);
   fChain->SetBranchAddress("fWebHistogram", &fWebHistogram, &b_event_fWebHistogram);
}

Bool_t sel01::Notify()
{
   // The Notify() function is called when a new file is opened. This
   // can be either for a new TTree in a TChain or when when a new TTree
   // is started when using PROOF. It is normaly not necessary to make changes
   // to the generated code, but the routine can be extended by the
   // user if needed. The return value is currently not used.

   return kTRUE;
}

#endif // #ifdef sel01_cxx
