//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Fri Jul 10 11:01:34 2009 by ROOT version 5.23/05
// from TTree t/t
// found on file: Memory Directory
//////////////////////////////////////////////////////////

#ifndef RooProofDriverSelector_h
#define RooProofDriverSelector_h

#include <TChain.h>
#include <TFile.h>
#include <TSelector.h>
class RooStudyPackage ;

class RooProofDriverSelector : public TSelector {
public :
   TTree          *fChain;   ///<!pointer to the analyzed TTree or TChain

   // Declaration of leaf types
   Int_t           i;

   // List of branches
   TBranch        *b_i;   ///<!

   RooProofDriverSelector(TTree * /*tree*/ =0) { b_i = 0 ; _pkg = 0 ; fChain = 0 ; }
   ~RooProofDriverSelector() override { }
   Int_t   Version() const override { return 2; }
   void    SlaveBegin(TTree *tree) override;
   void    Init(TTree* tree) override;
   bool  Notify() override;
   bool  Process(Long64_t entry) override;
   Int_t   GetEntry(Long64_t entry, Int_t getall = 0) override { return fChain ? fChain->GetTree()->GetEntry(entry, getall) : 0; }
   void    SetOption(const char *option) override { fOption = option; }
   void    SetObject(TObject *obj) override { fObject = obj; }
   void    SetInputList(TList *input) override { fInput = input; }
   void    SlaveTerminate() override ;
   TList  *GetOutputList() const override { return fOutput; }

   RooStudyPackage* _pkg ;
   Int_t      seed ;

   ClassDefOverride(RooProofDriverSelector,0);
};

#endif

