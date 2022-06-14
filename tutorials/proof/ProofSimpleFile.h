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
   virtual ~ProofSimpleFile();
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

   ClassDef(ProofSimpleFile,0);
};

#endif
