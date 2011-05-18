
//////////////////////////////////////////////////////////
//
// Example of TSelector implementation to do a Monte Carlo
// generation using Pythia8.
// See tutorials/proof/runProof.C, option "pythia8", for an
// example of how to run this selector.
//
//////////////////////////////////////////////////////////

#ifndef ProofPythia_h
#define ProofPythia_h

#include <TSelector.h>

class TClonesArray;
class TH1F;
class TPythia8;

class ProofPythia : public TSelector {
public :

   // Specific members
   TH1F            *fTot;
   TH1F            *fHist;
   TH1F            *fPt;
   TH1F            *fEta;
   TPythia8        *fPythia;
   TClonesArray    *fP;

   ProofPythia();
   virtual ~ProofPythia();
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

   ClassDef(ProofPythia,0);
};

#endif
