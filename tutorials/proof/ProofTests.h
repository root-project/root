/// \file
/// \ingroup tutorial_ProofTests
///
/// Auxilliary selector used to test PROOF functionality
///
/// \macro_code
///
/// \author Gerardo Ganis (gerardo.ganis@cern.ch)

#ifndef ProofTests_h
#define ProofTests_h

#include <TSelector.h>

class TH1I;

class ProofTests : public TSelector {
private:
   void            ParseInput();
public :

   // Specific members
   Int_t            fTestType;
   TH1I            *fStat;

   ProofTests();
   virtual ~ProofTests();
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

   ClassDef(ProofTests,0);
};

#endif
