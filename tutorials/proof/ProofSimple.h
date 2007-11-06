//////////////////////////////////////////////////////////
//
// Example of TSelector implementation to do generic
// processing (filling a set of histograms in this case)
//
//////////////////////////////////////////////////////////

#ifndef ProofSimple_h
#define ProofSimple_h

#include <TSelector.h>

class TH1D;
class TNtuple;

class ProofSimple : public TSelector {
public :

   // Specific members
   TH1D            *fHgaus;
   TH1D            *fHsqr;
   TNtuple         *fNtp;

   ProofSimple(TTree * /*tree*/ =0) { }
   virtual ~ProofSimple() { }
   virtual Int_t   Version() const { return 2; }
   virtual void    Begin(TTree *tree);
   virtual void    SlaveBegin(TTree *tree);
   virtual void    Init(TTree *tree);
   virtual Bool_t  Notify();
   virtual Bool_t  Process(Long64_t entry);
//   virtual Int_t   GetEntry(Long64_t, Int_t getall = 0) { return 0; }
   virtual void    SetOption(const char *option) { fOption = option; }
   virtual void    SetObject(TObject *obj) { fObject = obj; }
   virtual void    SetInputList(TList *input) { fInput = input; }
   virtual TList  *GetOutputList() const { return fOutput; }
   virtual void    SlaveTerminate();
   virtual void    Terminate();

   ClassDef(ProofSimple,0);
};

#endif

#ifdef ProofSimple_cxx
void ProofSimple::Init(TTree *)
{
   // The Init() function is called when the selector needs to initialize
   // a new tree or chain. Typically here the branch addresses and branch
   // pointers of the tree will be set.
   // It is normaly not necessary to make changes to the generated
   // code, but the routine can be extended by the user if needed.
   // Init() will be called many times when running on PROOF
   // (once per file to be processed).

}

Bool_t ProofSimple::Notify()
{
   // The Notify() function is called when a new file is opened. This
   // can be either for a new TTree in a TChain or when when a new TTree
   // is started when using PROOF. It is normaly not necessary to make changes
   // to the generated code, but the routine can be extended by the
   // user if needed. The return value is currently not used.

   return kTRUE;
}

#endif // #ifdef ProofSimple_cxx
