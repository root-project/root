/// \file
/// \ingroup tutorial_proofntuple
///
/// Selector to fill a simple ntuple
///
/// \macro_code
///
/// \author Gerardo Ganis (gerardo.ganis@cern.ch)

#ifndef ProofNtuple_h
#define ProofNtuple_h

#include <TSelector.h>

class TFile;
class TProofOutputFile;
class TNtuple;
class TRandom3;

class ProofNtuple : public TSelector {
public :

   // Specific members
   TFile            *fFile;
   TProofOutputFile *fProofFile; // For optimized merging of the ntuple
   TNtuple          *fNtp;
   TNtuple          *fNtp2;      // To test double TTree in the same file
   TRandom3         *fRandom;
   Bool_t            fPlotNtuple;
   TNtuple          *fNtpRndm;   // Ntuple with random numbers

   ProofNtuple() : fFile(0), fProofFile(0), fNtp(0), fNtp2(0), fRandom(0), fPlotNtuple(kTRUE), fNtpRndm(0)  { }
   virtual ~ProofNtuple();
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

   void PlotNtuple(TNtuple *, const char *);

   ClassDef(ProofNtuple,0);
};

#endif
