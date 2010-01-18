//////////////////////////////////////////////////////////
//
// Example of TSelector implementation to do generic
// processing (filling a set of histograms in this case).
// See tutorials/proof/runProof.C, option "simple", for an
// example of how to run this selector.
//
//////////////////////////////////////////////////////////

#ifndef ProofSimple_h
#define ProofSimple_h

#include <TSelector.h>

class TH1F;
class TRandom3;

class ProofSimple : public TSelector {
public :

   // Specific members
   Int_t            fNhist;
   TH1F           **fHist;//[fNhist]
   TRandom3        *fRandom;

   ProofSimple();
   virtual ~ProofSimple();
   virtual Int_t   Version() const { return 2; }
   virtual void    Begin(TTree *tree);
   virtual void    SlaveBegin(TTree *tree);
   virtual Bool_t  Process(Long64_t entry);
   virtual void    SetOption(const char *option) { fOption = option; }
   virtual void    SetObject(TObject *obj) { fObject = obj; }
   virtual void    SetInputList(TList *input) { fInput = input; }
   virtual TList  *GetOutputList() const { return fOutput; }
   virtual void    SlaveTerminate();
   virtual void    Terminate();

   ClassDef(ProofSimple,0);
};

#endif
