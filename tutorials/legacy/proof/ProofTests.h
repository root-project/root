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
   ~ProofTests() override;
   Int_t   Version() const override { return 2; }
   void    Begin(TTree *tree) override;
   void    SlaveBegin(TTree *tree) override;
   Bool_t  Process(Long64_t entry) override;
   void    SetOption(const char *option) override { fOption = option; }
   void    SetObject(TObject *obj) override { fObject = obj; }
   void    SetInputList(TList *input) override { fInput = input; }
   TList  *GetOutputList() const override { return fOutput; }
   void    SlaveTerminate() override;
   void    Terminate() override;

   ClassDefOverride(ProofTests,0);
};

#endif
