//////////////////////////////////////////////////////////
//   This class has been automatically generated 
//     (Wed Oct 23 14:06:49 2002 by ROOT version3.03/09)
//   from TTree ntuple/Demo ntuple
//   found on file: hsimple.root
//////////////////////////////////////////////////////////


#ifndef mc01_h
#define mc01_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TProxy.h>

class mc01 {
   public :
   TTree          *fChain;   //!pointer to the analyzed TTree or TChain
   Int_t           fCurrent; //!current Tree number in a TChain
   TProxyDirector  fDirector; //!object shared by the proxies.

//Declaration of leaves types
   TFloatProxy     px;
   TFloatProxy     py;
   TFloatProxy     pz;
   TFloatProxy     random;
   TFloatProxy     i;

   mc01(TTree *tree=0);
   ~mc01();
   Int_t  Cut(Int_t entry);
   Int_t  GetEntry(Int_t entry);
   Int_t  LoadTree(Int_t entry);
   void   Init(TTree *tree);
   void   Loop(int arg = 0);
   Bool_t Notify();
   void   Show(Int_t entry = -1);
};

#endif

#ifdef mc01_cxx
mc01::mc01(TTree *tree) :
   fDirector(tree,-1),
      px(&fDirector,"px"),
      py(&fDirector,"py"),
      pz(&fDirector,"pz"),
      random(&fDirector,"random"),
      i(&fDirector,"i")
{
// if parameter tree is not specified (or zero), connect the file
// used to generate this class and read the Tree.
   if (tree == 0) {
      TFile *f = (TFile*)gROOT->GetListOfFiles()->FindObject("hsimple.root");
      if (!f) {
         f = new TFile("hsimple.root");
      }
      tree = (TTree*)gDirectory->Get("ntuple");

   }
   Init(tree);
}

mc01::~mc01()
{
   if (!fChain) return;
   //delete fChain->GetCurrentFile();
}

Int_t mc01::GetEntry(Int_t entry)
{
// Read contents of entry.
   if (!fChain) return 0;
   return fChain->GetEntry(entry);
}
Int_t mc01::LoadTree(Int_t entry)
{
// Set the environment to read one entry
   if (!fChain) return -5;
   Int_t centry = fChain->LoadTree(entry);
   if (centry < 0) return centry;
   if (fChain->IsA() != TChain::Class()) return centry;
   TChain *chain = (TChain*)fChain;
   if (chain->GetTreeNumber() != fCurrent) {
      fCurrent = chain->GetTreeNumber();
      Notify();
   }
   return centry;
}

void mc01::Init(TTree *tree)
{
//   Set branch addresses
   if (tree == 0) return;
   fChain    = tree;
   fCurrent = -1;
   fChain->SetMakeClass(1);

   Notify();
}

Bool_t mc01::Notify()
{
   // Called when loading a new file.
   // Get branch pointers.
   fDirector.fTree = fChain;
   return kTRUE;
}

void mc01::Show(Int_t entry)
{
// Print contents of entry.
// If entry is not specified, print current entry
   if (!fChain) return;
   fChain->Show(entry);
}
Int_t mc01::Cut(Int_t entry)
{
// This function may be called from Loop.
// returns  1 if entry is accepted.
// returns -1 otherwise.
   return 1;
}
#endif // #ifdef mc01_cxx

