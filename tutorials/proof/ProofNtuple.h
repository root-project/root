/// \file
/// \ingroup tutorial_legacy
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

   ProofNtuple() : fFile(nullptr), fProofFile(nullptr), fNtp(nullptr), fNtp2(nullptr), fRandom(nullptr), fPlotNtuple(kTRUE), fNtpRndm(nullptr)  { }
   ~ProofNtuple() override;
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

   void PlotNtuple(TNtuple *, const char *);

   ClassDefOverride(ProofNtuple,0);
};

#endif
