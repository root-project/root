//////////////////////////////////////////////////////////
//
// Example of TSelector implementation to do generic processing
// (filling a set of histograms in this case) and merging via
// a file, with part of the objects saved in a sub-directory.
// See tutorials/proof/runProof.C, option "simplefile", for an
// example of how to run this selector.
//
//////////////////////////////////////////////////////////

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
