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
   virtual ~ProofSimple();
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

   void            FillNtuple(Long64_t entry);
   void            PlotNtuple(TNtuple *, const char *);
   Int_t           GetHistosFromFC(TCanvas *);

   // Setters and getters (for TDataMember)
   Int_t GetNhist() { return fNhist; }
   void SetNhist(Int_t nh) { fNhist = nh; }
   Int_t GetNhist3() { return fNhist3; }
   void SetNhist3(Int_t nh) { fNhist3 = nh; }

   ClassDef(ProofSimple,3);
};

#endif
