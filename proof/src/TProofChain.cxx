// @(#)root/proof:$Name:  $:$Id:$
// Author: G. Ganis  Nov 2006

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProofChain                                                          //
//                                                                      //
// A TChain proxy on PROOF.                                             //
// Uses an internal TDSet to handle processing.                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TProofChain.h"
#include "TDSet.h"
#include "TVirtualProof.h"


ClassImp(TProofChain)

//______________________________________________________________________________
TProofChain::TProofChain() : TChain()
{
   // Crates a new Proof chain proxy containing the files from the TDSet.

   fChain        = 0;
   fTree         = 0;
   fSet          = 0;
   fDirectory    = gDirectory;
   fDrawFeedback = 0;
}

//______________________________________________________________________________
TProofChain::TProofChain(TChain *chain) : TChain()
{
   // Crates a new Proof chain proxy containing the files from the TDSet.

   fChain        = chain;
   fTree         = 0;
   fSet          = chain ? new TDSet((const TChain &)(*chain)) : 0;
   fDirectory    = gDirectory;
   fDrawFeedback = 0;
   if (gProof)
      gProof->AddChain(chain);
}

//______________________________________________________________________________
TProofChain::~TProofChain()
{
   // Destructor

   if (fChain) {
      if (fChain)
         gProof->RemoveChain(fChain);
      SafeDelete(fSet);
   } else {
      // Not owner
      fSet = 0;
   }
   SafeDelete(fTree);
   fDirectory    = 0;
}

//______________________________________________________________________________
TProofChain::TProofChain(TDSet *dset, Bool_t gettreeheader)
{
   // Constructor from existing data set

   fChain        = 0;
   fTree         = 0;
   fSet          = dset;
   fDirectory    = gDirectory;
   fDrawFeedback = 0;
   if (gProof) {
      ConnectProof();
      if (gettreeheader && dset)
         fTree = gProof->GetTreeHeader(dset);
   }
}

//______________________________________________________________________________
void TProofChain::Browse(TBrowser *b)
{
   // Forwards the execution to the dummy tree header.
   // See TTree::Browse().

   fSet->Browse(b);
}

//______________________________________________________________________________
Long64_t TProofChain::Draw(const char *varexp, const TCut &selection,
                           Option_t *option, Long64_t nentries, Long64_t firstentry)
{
   // Forwards the execution to the TDSet.
   // Returns -1 in case of error or number of selected events in case of success.
   // See TDSet::Browse().

   if (!gProof) {
      Error("Draw", "no active PROOF session");
      return -1;
   }
   ConnectProof();

   if (fDrawFeedback)
      gProof->SetDrawFeedbackOption(fDrawFeedback, option);
   fReadEntry = firstentry;
   fSet->SetEventList(fEventList);

   Long64_t rv = fSet->Draw(varexp, selection, option, nentries, firstentry);
   return rv;
}

//______________________________________________________________________________
Long64_t TProofChain::Draw(const char *varexp, const char *selection,
                           Option_t *option,Long64_t nentries, Long64_t firstentry)
{
   // Forwards the execution to the TDSet.
   // Returns -1 in case of error or number of selected events in case of success.
   // See TDSet::Browse().

   if (!gProof) {
      Error("Draw", "no active PROOF session");
      return -1;
   }
   ConnectProof();

   if (fDrawFeedback)
      gProof->SetDrawFeedbackOption(fDrawFeedback, option);
   fReadEntry = firstentry;
   fSet->SetEventList(fEventList);
   Long64_t rv = fSet->Draw(varexp, selection, option, nentries, firstentry);
   return rv;
}

//______________________________________________________________________________
TBranch *TProofChain::FindBranch(const char* branchname)
{
   // Forwards the execution to the dummy tree header.
   // See TTree::FindBranch().

   return (fTree ? fTree->FindBranch(branchname) : (TBranch *)0);
}

//______________________________________________________________________________
TLeaf *TProofChain::FindLeaf(const char* searchname)
{
   // Forwards the execution to the dummy tree header.
   // See TTree::FindLeaf().

   return (fTree ? fTree->FindLeaf(searchname) : (TLeaf *)0);
}

//______________________________________________________________________________
TBranch *TProofChain::GetBranch(const char *name)
{
   // Forwards the execution to the dummy tree header.
   // See TTree::GetBranch().

   return (fTree ? fTree->GetBranch(name) : (TBranch *)0);
}

//______________________________________________________________________________
Bool_t TProofChain::GetBranchStatus(const char *branchname) const
{
   // Forwards the execution to the dummy tree header.
   // See TTree::GetBranchStatus().

   return (fTree ? fTree->GetBranchStatus(branchname) : kFALSE);
}

//______________________________________________________________________________
Int_t TProofChain::GetBranchStyle()
{
   // See TTree::GetBranchStyle().

   return fgBranchStyle;
}

//______________________________________________________________________________
TVirtualTreePlayer *TProofChain::GetPlayer()
{
   // Forwards the execution to the dummy tree header.
   // See TTree::GetPlayer().

   if (!fTree)
      if (gProof) {
         fTree = gProof->GetTreeHeader(fSet);
         ConnectProof();
      }

   return (fTree ? fTree->GetPlayer() : (TVirtualTreePlayer *)0);
}

//______________________________________________________________________________
Long64_t TProofChain::Process(const char *filename, Option_t *option,
                              Long64_t nentries, Long64_t firstentry)
{
   // Forwards the execution to the TDSet.
   // The return value is -1 in case of error and TSelector::GetStatus() in
   // in case of success.
   // See TDSet::Process().

   return fSet->Process(filename, option, nentries, firstentry);
}

//______________________________________________________________________________
Long64_t TProofChain::Process(TSelector *selector, Option_t *option,
                              Long64_t nentries, Long64_t firstentry)
{
   // Not implemented in TProofChain. Shouldn't be used.
   // The return value is -1 in case of error and TSelector::GetStatus() in
   // in case of success.

   if (selector || option || nentries || firstentry);
   //   return fSet->Process(selector, option, nentries, firstentry);
   Warning("Process", "not implemented"); // TODO
   return -1;
}

//______________________________________________________________________________
void TProofChain::SetDebug(Int_t level, Long64_t min, Long64_t max)
{
   // See TTree::SetDebug

   TTree::SetDebug(level, min, max);
}

//______________________________________________________________________________
void TProofChain::SetName(const char *name)
{
   // See TTree::GetName.

   TTree::SetName(name);
}

//______________________________________________________________________________
Long64_t TProofChain::GetEntries() const
{
   // Returns the total number of entries in the TProofChain, which is
   // the number of entries in the TDSet that it holds.

   // this was used for holding the total number of entries
   return (fTree ? fTree->GetMaxEntryLoop() : (Long64_t)(-1));
}

//______________________________________________________________________________
Long64_t TProofChain::GetEntries(const char *)
{
   // See TTree::GetEntries(const char *selection)
   // Not implemented in TProofChain. Shouldn't be used.

   return Long64_t(-1);
}

//______________________________________________________________________________
void TProofChain::Progress(Long64_t total, Long64_t processed)
{
   // Changes the number of processed entries.

   if (gROOT->IsInterrupted() && gProof)
      gProof->StopProcess(kTRUE);
   if (total)
      ;

   fReadEntry = processed;
}

//______________________________________________________________________________
Long64_t TProofChain::GetReadEntry() const
{
   // Returns the number of processed entries.

   return fReadEntry;
}

//______________________________________________________________________________
void TProofChain::ReleaseProof()
{
   // Releases PROOF. Disconnect the "progress" signal.

   if (!gProof)
      return;
   gProof->Disconnect("Progress(Long64_t,Long64_t)",
                      this, "Progress(Long64_t,Long64_t)");
   if (fDrawFeedback)
      gProof->DeleteDrawFeedback(fDrawFeedback);
   fDrawFeedback = 0;
}

//______________________________________________________________________________
void TProofChain::ConnectProof()
{
   // Connects the proof - creates a "DrawFeedback" and connects the
   // "Progress" signal.

   if (gProof && !fDrawFeedback) {
      fDrawFeedback = gProof->CreateDrawFeedback();

      gProof->Connect("Progress(Long64_t,Long64_t)", "TProofChain",
                       this, "Progress(Long64_t,Long64_t)");
   }
}
