//////////////////////////////////////////////////////////
//   This class has been automatically generated 
//     (Wed Oct 23 14:06:51 2002 by ROOT version3.03/09)
//   from TTree ntuple/Demo ntuple
//   found on file: hsimple.root
//////////////////////////////////////////////////////////


#ifndef mc02_h
#define mc02_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>

class mc02 {
   public :
   TTree          *fChain;   //!pointer to the analyzed TTree or TChain
   Int_t           fCurrent; //!current Tree number in a TChain
//Declaration of leaves types
   Float_t         px;
   Float_t         py;
   Float_t         pz;
   Float_t         random;
   Float_t         i;

//List of branches
   TBranch        *b_px;   //!
   TBranch        *b_py;   //!
   TBranch        *b_pz;   //!
   TBranch        *b_random;   //!
   TBranch        *b_i;   //!

   mc02(TTree *tree=0);
   ~mc02();
   Int_t  Cut(Int_t entry);
   Int_t  GetEntry(Int_t entry);
   Int_t  LoadTree(Int_t entry);
   void   Init(TTree *tree);
   void   Loop(int arg = 100);
   Bool_t Notify();
   void   Show(Int_t entry = -1);
};

#endif

#ifdef mc02_cxx
mc02::mc02(TTree *tree)
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

mc02::~mc02()
{
   if (!fChain) return;
   //delete fChain->GetCurrentFile();
}

Int_t mc02::GetEntry(Int_t entry)
{
// Read contents of entry.
   if (!fChain) return 0;
   return fChain->GetEntry(entry);
}
Int_t mc02::LoadTree(Int_t entry)
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

void mc02::Init(TTree *tree)
{
//   Set branch addresses
   if (tree == 0) return;
   fChain    = tree;
   fCurrent = -1;
   fChain->SetMakeClass(1);

   fChain->SetBranchAddress("px",&px);
   fChain->SetBranchAddress("py",&py);
   fChain->SetBranchAddress("pz",&pz);
   fChain->SetBranchAddress("random",&random);
   fChain->SetBranchAddress("i",&i);
   Notify();
}

Bool_t mc02::Notify()
{
   // Called when loading a new file.
   // Get branch pointers.
   b_px = fChain->GetBranch("px");
   b_py = fChain->GetBranch("py");
   b_pz = fChain->GetBranch("pz");
   b_random = fChain->GetBranch("random");
   b_i = fChain->GetBranch("i");
   return kTRUE;
}

void mc02::Show(Int_t entry)
{
// Print contents of entry.
// If entry is not specified, print current entry
   if (!fChain) return;
   fChain->Show(entry);
}
Int_t mc02::Cut(Int_t entry)
{
// This function may be called from Loop.
// returns  1 if entry is accepted.
// returns -1 otherwise.
   return 1;
}
#endif // #ifdef mc02_cxx

