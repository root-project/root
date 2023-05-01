//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Fri Apr 23 20:17:55 2010 by ROOT version 5.22/00
// from TTree physics/physics
// found on file: ZeeNp5.wzD3PD.root
//////////////////////////////////////////////////////////

#ifndef emptysel_h
#define emptysel_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TSelector.h>
#include <iostream>

class emptysel : public TSelector {
public :

  int m_testvar;
  void printAddress() { 
    // cout << "This = " << this << std::endl;i
  };

  TTree          *fChain;   //!pointer to the analyzed TTree or TChain

 emptysel(TTree * /*tree*/ =0): m_testvar(-999) { 
     std::cout << "In constructor" << endl;
     printAddress();
     std::cout << "Testvar in constructor: " << m_testvar 
	       << std::endl << std::endl;
   };

   ~emptysel() override { }
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

   ClassDefOverride(emptysel,0);
};

#endif

#ifdef emptysel_cxx
void emptysel::Init(TTree *tree)
{
   // The Init() function is called when the selector needs to initialize
   // a new tree or chain. Typically here the branch addresses and branch
   // pointers of the tree will be set.
   // It is normally not necessary to make changes to the generated
   // code, but the routine can be extended by the user if needed.
   // Init() will be called many times when running on PROOF
   // (once per file to be processed).

   // Set object pointer
   // Set branch addresses and branch pointers
   if (!tree) return;
   fChain = tree;
   fChain->SetMakeClass(1);

}

Bool_t emptysel::Notify()
{
   // The Notify() function is called when a new file is opened. This
   // can be either for a new TTree in a TChain or when when a new TTree
   // is started when using PROOF. It is normally not necessary to make changes
   // to the generated code, but the routine can be extended by the
   // user if needed. The return value is currently not used.

   return kTRUE;
}

#endif // #ifdef emptysel_cxx
