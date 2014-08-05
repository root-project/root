//////////////////////////////////////////////////////////
//
// Example of TSelector implementation to do generic
// processing with stdlib collections.
// See tutorials/proof/runProof.C, option "stdlib", for an
// example of how to run this selector.
//
//////////////////////////////////////////////////////////

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
   virtual ~ProofStdVect();
   virtual Int_t   Version() const { return 2; }
   virtual void    Begin(TTree *tree);
   virtual void    SlaveBegin(TTree *tree);
   void    Init(TTree *tree);
   Bool_t  Notify();
   virtual Bool_t  Process(Long64_t entry);
   virtual Int_t   GetEntry(Long64_t entry, Int_t getall = 0) { return fChain ? fChain->GetTree()->GetEntry(entry, getall) : 0; }
   virtual void    SetOption(const char *option) { fOption = option; }
   virtual void    SetObject(TObject *obj) { fObject = obj; }
   virtual void    SetInputList(TList *input) { fInput = input; }
   virtual TList  *GetOutputList() const { return fOutput; }
   virtual void    SlaveTerminate();
   virtual void    Terminate();

   ClassDef(ProofStdVect,0);
};

#endif
