//////////////////////////////////////////////////////////
//   This class has been automatically generated 
//     (Tue Oct 22 14:14:28 2002 by ROOT version3.03/09)
//   from TTree ntuple/Demo ntuple
//   found on file: hsimple.root
//////////////////////////////////////////////////////////


#ifndef test01_h
#define test01_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TSelector.h>
#include "TProxy.h"

class test01 : public TSelector {
   public :
   TTree          *fChain;   //!pointer to the analyzed TTree or TChain
//Declaration of leaves types
   TFloatProxy     px;
   TFloatProxy     py;
   TFloatProxy     pz;
   TFloatProxy     random;
   TFloatProxy     i;

//List of branches
   test01(TTree *tree=0) :
      px(tree,"px"),
      py(tree,"py"),
      pz(tree,"pz"),
      random(tree,"random"),
      i(tree,"i")
      { }
   ~test01() { }
   void    Begin(TTree *tree);
   void    Init(TTree *tree);
   Bool_t  Notify();
   Bool_t  Process(Int_t entry);
   Bool_t  ProcessCut(Int_t entry);
   void    ProcessFill(Int_t entry);
   void    SetOption(const char *option) { fOption = option; }
   void    SetObject(TObject *obj) { fObject = obj; }
   void    SetInputList(TList *input) {fInput = input;}
   TList  *GetOutputList() const { return fOutput; }
   void    Terminate();
#include "code.C"

};

#endif

#ifdef test01_cxx
void test01::Init(TTree *tree)
{
//   Set branch addresses
   if (tree == 0) return;
   fChain    = tree;
   // fChain->SetMakeClass(1);

   Notify();
}

Bool_t test01::Notify()
{
   // Called when loading a new file.
   // Get branch pointers.
   px.SetTree(fChain);
   py.SetTree(fChain);
   pz.SetTree(fChain);
   random.SetTree(fChain);
   i.SetTree(fChain);
   return kTRUE;
}

#endif // #ifdef test01_cxx

