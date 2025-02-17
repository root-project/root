/// \file
/// \ingroup tutorial_ProofSimple
///
/// Selector to fill a set of histograms
///
/// \macro_code
///
/// \author Gerardo Ganis (gerardo.ganis@cern.ch)

#ifndef ProofSimple_h
#define ProofSimple_h

#include <TSelector.h>

class TH1F;
class TH3F;
class TFile;
class TProofOutputFile;
class TNtuple;
class TRandom3;
class TCanvas;

class ProofSimple : public TSelector {
public :

   // Specific members
   Int_t             fNhist;
   TH1F            **fHist;//![fNhist]
   Int_t             fNhist3;
   TH3F            **fHist3;//![fNhist3]
   TFile            *fFile;
   TProofOutputFile *fProofFile; // For optimized merging of the ntuple
   TNtuple          *fNtp;
   Bool_t            fPlotNtuple;
   Int_t             fHasNtuple;
   TRandom3         *fRandom;//!

   TH1F             *fHLab;//!

   ProofSimple();
   ~ProofSimple() override;
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

   void            FillNtuple(Long64_t entry);
   void            PlotNtuple(TNtuple *, const char *);
   Int_t           GetHistosFromFC(TCanvas *);

   // Setters and getters (for TDataMember)
   Int_t GetNhist() { return fNhist; }
   void SetNhist(Int_t nh) { fNhist = nh; }
   Int_t GetNhist3() { return fNhist3; }
   void SetNhist3(Int_t nh) { fNhist3 = nh; }

   ClassDefOverride(ProofSimple,3);
};

#endif
