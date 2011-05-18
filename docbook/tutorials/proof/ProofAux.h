//////////////////////////////////////////////////////////////
//
// Selector used for auxilliary actions in the PROOF tutorials
//
//////////////////////////////////////////////////////////////

#ifndef ProofAux_h
#define ProofAux_h

#include <TSelector.h>

class TList;

class ProofAux : public TSelector {
private :
   Int_t           GenerateTree(const char *fnt, Long64_t ent, TString &fn);
   Int_t           GenerateFriend(const char *fnt,  const char *fnf = 0);
   Int_t           GetAction(TList *input);
public :

   // Specific members
   Int_t           fAction;
   Long64_t        fNEvents;
   TList          *fMainList;
   TList          *fFriendList;

   ProofAux();
   virtual ~ProofAux();
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

   ClassDef(ProofAux,0);
};

#endif
