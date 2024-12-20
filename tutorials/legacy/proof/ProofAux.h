/// \file
/// \ingroup tutorial_ProofAux
///
/// Selector used for auxiliary actions in the PROOF tutorials
///
/// \macro_code
///
/// \author Gerardo Ganis (gerardo.ganis@cern.ch)

#ifndef ProofAux_h
#define ProofAux_h

#include <TString.h>
#include <TSelector.h>

class TList;

class ProofAux : public TSelector {
private :
   Int_t           GenerateTree(const char *fnt, Long64_t ent, TString &fn);
   Int_t           GenerateFriend(const char *fnt,  const char *fnf = nullptr);
   Int_t           GetAction(TList *input);
public :

   // Specific members
   Int_t           fAction;
   Long64_t        fNEvents;
   TList          *fMainList;
   TList          *fFriendList;
   TString         fDir;

   ProofAux();
   ~ProofAux() override;
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

   ClassDefOverride(ProofAux,0);
};

#endif
