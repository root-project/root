/// \file
/// \ingroup tutorial_ProofSimpleFile
///
/// Selector to fill a set of histograms and merging via file
///
/// \macro_code
///
/// \author Gerardo Ganis (gerardo.ganis@cern.ch)

#ifndef ProofSimpleFile_h
#define ProofSimpleFile_h

#include <TSelector.h>

class TH1F;
class TRandom3;
class TFile;
class TProofOutputFile;
class TDirectory;

class ProofSimpleFile : public TSelector {
private:
   Int_t CreateHistoArrays();
   void PlotHistos(Int_t opt = 0);
public :

   // Specific members
   Int_t             fNhist;
   TH1F            **fHistTop;//[fNhist]
   TH1F            **fHistDir;//[fNhist]
   TRandom3         *fRandom;
   TFile            *fFile;
   TProofOutputFile *fProofFile; // For merging via file
   TDirectory       *fFileDir;   // Subdirectory for some histos

   ProofSimpleFile();
   ~ProofSimpleFile() override;
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

   ClassDefOverride(ProofSimpleFile,0);
};

#endif
