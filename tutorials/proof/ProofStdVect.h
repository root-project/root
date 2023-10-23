/// \file
/// \ingroup tutorial_ProofStdVec
///
/// Selector for generic processing with stdlib collections
///
/// \macro_code
///
/// \author Gerardo Ganis (gerardo.ganis@cern.ch)

#ifndef ProofStdVect_h
#define ProofStdVect_h

#include <TSelector.h>
#include <TChain.h>

#include <vector>
#ifdef __MAKECINT__
#pragma link C++ class std::vector<std::vector<bool> >+;
#pragma link C++ class std::vector<std::vector<float> >+;
#endif

class TFile;
class TProofOutputFile;
class TTree;
class TRandom3;
class TH1F;

class ProofStdVect : public TSelector {
public :

   // Specific members
   Bool_t            fCreate; //! True if in create files mode

   // Create mode
   TTree            *fTree;   //! The tree filled in create mode
   TFile            *fFile;   //! Output file in create mode
   TProofOutputFile *fProofFile; //! For dataset creation in create mode
   TRandom3         *fRandom; //! Random generator in create mode
   TH1F             *fHgood;   //! Histogram with good hits
   TH1F             *fHbad;   //! Histogram with bad hits
   // Std vector members
   std::vector<std::vector<bool> > fVb; //! Booleans
   std::vector<std::vector<float> > fVfx; //! Floats x
   std::vector<std::vector<float> > fVfy; //! Floats y

   // Read mode
   TTree          *fChain;   //!pointer to the analyzed TTree or TChain
   // Declaration of leaf types
   std::vector<std::vector<bool> > *fVbr; //!
   std::vector<std::vector<float> > *fVfxr; //!
   std::vector<std::vector<float> > *fVfyr; //!
   // List of branches
   TBranch        *b_Vb;   //!
   TBranch        *b_Vfx;   //!
   TBranch        *b_Vfy;   //!

   ProofStdVect();
   ~ProofStdVect() override;
   Int_t   Version() const override { return 2; }
   void    Begin(TTree *tree) override;
   void    SlaveBegin(TTree *tree) override;
   void    Init(TTree *tree) override;
   Bool_t  Notify() override;
   Bool_t  Process(Long64_t entry) override;
   Int_t   GetEntry(Long64_t entry, Int_t getall = 0) override { return fChain ? fChain->GetTree()->GetEntry(entry, getall) : 0; }
   void    SetOption(const char *option) override { fOption = option; }
   void    SetObject(TObject *obj) override { fObject = obj; }
   void    SetInputList(TList *input) override { fInput = input; }
   TList  *GetOutputList() const override { return fOutput; }
   void    SlaveTerminate() override;
   void    Terminate() override;

   ClassDefOverride(ProofStdVect,0);
};

#endif
