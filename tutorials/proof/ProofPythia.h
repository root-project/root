/// \file
/// \ingroup tutorial_legacy
///
/// Selector to generate Monte Carlo events with Pythia8
///
/// \macro_code
///
/// \author Gerardo Ganis (gerardo.ganis@cern.ch)

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
   ~ProofPythia() override;
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

   ClassDefOverride(ProofPythia,0);
};

#endif
