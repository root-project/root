/// \file
/// \ingroup tutorial_ProofEvent
///
/// Selector for generic processing with Event
///
/// \macro_code
///
/// \author Gerardo Ganis (gerardo.ganis@cern.ch)

#ifndef ProofEvent_h
#define ProofEvent_h

#include <TSelector.h>
#include "Event.h"

class TH1F;
class TRandom3;

class ProofEvent : public TSelector {
public :

   // Specific members
   TRandom3        *fRandom;
   Event           *fEvent;
   Int_t            fNtrack;
   TH1F            *fHisto;

   ProofEvent();
   virtual ~ProofEvent();
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

   ClassDef(ProofEvent,0);
};

#endif
