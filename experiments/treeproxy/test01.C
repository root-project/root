#define test01_cxx
// The class definition in test01.h has been generated automatically
// by the ROOT utility TTree::MakeSelector().
//
// This class is derived from the ROOT class TSelector.
// The following members functions are called by the TTree::Process() functions:
//    Begin():       called everytime a loop on the tree starts,
//                   a convenient place to create your histograms.
//    Notify():      this function is called at the first entry of a new Tree
//                   in a chain.
//    ProcessCut():  called at the beginning of each entry to return a flag,
//                   true if the entry must be analyzed.
//    ProcessFill(): called in the entry loop for all entries accepted
//                   by Select.
//    Terminate():   called at the end of a loop on the tree,
//                   a convenient place to draw/fit your histograms.
//
//   To use this file, try the following session on your Tree T
//
// Root > T->Process("test01.C")
// Root > T->Process("test01.C","some options")
// Root > T->Process("test01.C+")
//
#include "test01.h"
#include "TH2.h"
#include "TStyle.h"
#include "TCanvas.h"


void test01::Begin(TTree *tree)
{
   // Function called before starting the event loop.
   // Initialize the tree branches.

   Init(tree);

   TString option = GetOption();

}

Bool_t test01::Process(Int_t entry)
{
   // Processing function.
   // Entry is the entry number in the current tree.
   // Read only the necessary branches to select entries.
   // To read complete event, call fChain->GetTree()->GetEntry(entry).
   // Return kFALSE to stop processing.

   return kTRUE;
}

Bool_t test01::ProcessCut(Int_t entry)
{
   // Selection function.
   // Entry is the entry number in the current tree.
   // Read only the necessary branches to select entries.
   // Return kFALSE as soon as a bad entry is detected.
   // To read complete event, call fChain->GetTree()->GetEntry(entry).

   return kTRUE;
}

void test01::ProcessFill(Int_t entry)
{
   // Function called for selected entries only.
   // Entry is the entry number in the current tree.
   // Read branches not processed in ProcessCut() and fill histograms.
   // To read complete event, call fChain->GetTree()->GetEntry(entry).

   // fChain->GetTree()->GetEntry();
   code();
}

void test01::Terminate()
{
   // Function called at the end of the event loop.

}
