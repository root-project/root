// @(#)root/tree:$Name:  $:$Id: TTree.cxx,v 1.216 2004/12/01 15:45:27 rdm Exp $
// Author: Marek Biskup   10/3/2005

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
// A wrapper for TDSet to behave as a Tree/Chain.                       //
// Uses an internal TDSet to handle processing and a TTree              //
// which holds the branch structure.                                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TProofChain.h"
#include "TDSet.h"
#include "TVirtualProof.h"
#include "TDrawFeedback.h"

ClassImp(TProofChain)

//______________________________________________________________________________
TProofChain::TProofChain(TDSet *set, TTree *tree) : TTree()
{
   // Crates a new TProof chain containing the files from the TDSet.
   // The tree is just a dummy containing descriptions of all the tree leaves.

   fTree      = tree;
   fSet       = set;
   fDirectory = gDirectory;
   fProof     = 0;
}

//______________________________________________________________________________
TProofChain::~TProofChain()
{
   // Destructor - removes the chain from the proof in case a proof was set.

   ReleaseProof();
   delete fTree;
}

//______________________________________________________________________________
void TProofChain::AddClone(TTree *)
{
   // Not implemented in TProofChain. Shouldn't be used.

   Error("TProofChain::AddClone","not implemented");
}

//______________________________________________________________________________
TFriendElement *TProofChain::AddFriend(const char *, const char *)
{
   // Not implemented in TProofChain. Shouldn't be used.

   return 0;
}
//______________________________________________________________________________
TFriendElement *TProofChain::AddFriend(const char *, TFile *)
{
   // Not implemented in TProofChain. Shouldn't be used.

   return 0;

}
//______________________________________________________________________________
TFriendElement *TProofChain::AddFriend(TTree *, const char*, Bool_t)
{
   // Not implemented in TProofChain. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
Long64_t TProofChain::AutoSave(Option_t *)
{
   // Not implemented in TProofChain. Shouldn't be used.

   Error("TProofChain::AutoSave","not implemented");
   return 0;
}

//______________________________________________________________________________
Int_t TProofChain::Branch(TList *, Int_t, Int_t)
{
   // Not implemented in TProofChain. Shouldn't be used.

   Error("TProofChain::Branch","not implemented");
   return 0;
}

//______________________________________________________________________________
Int_t TProofChain::Branch(TCollection *, Int_t, Int_t, const char *)
{
   // Not implemented in TProofChain. Shouldn't be used.

   Error("TProofChain::Branch","not implemented");
   return 0;
}

//______________________________________________________________________________
Int_t TProofChain::Branch(const char *, Int_t, Int_t)
{
   // Not implemented in TProofChain. Shouldn't be used.

   Error("TProofChain::Branch","not implemented");
   return 0;
}

//______________________________________________________________________________
TBranch *TProofChain::Branch(const char *, TClonesArray **, Int_t, Int_t)
{
   // Not implemented in TProofChain. Shouldn't be used.

   Error("TProofChain::Branch","not implemented");
   return 0;
}

//______________________________________________________________________________
TBranch *TProofChain::Branch(const char *, void *, const char *, Int_t)
{
   // Not implemented in TProofChain. Shouldn't be used.

   Error("TProofChain::Branch","not implemented");
   return 0;
}

//______________________________________________________________________________
TBranch *TProofChain::Branch(const char *, void *, Int_t, Int_t)
{
   // Not implemented in TProofChain. Shouldn't be used.

   Error("TProofChain::Branch","not implemented");
   return 0;
}

//______________________________________________________________________________
TBranch *TProofChain::Branch(const char *, const char *, void *, Int_t, Int_t)
{
   // Not implemented in TProofChain. Shouldn't be used.

   Error("TProofChain::Branch","not implemented");
   return 0;
}

//______________________________________________________________________________
TBranch *TProofChain::BranchOld(const char *, const char *, void *, Int_t, Int_t)
{
   // Not implemented in TProofChain. Shouldn't be used.

   Error("TProofChain::BranchOld","not implemented");
   return 0;
}

//______________________________________________________________________________
TBranch *TProofChain::BranchRef()
{
   // Not implemented in TProofChain. Shouldn't be used.

   Error("TProofChain::BranchRef", "not implemented");
   return 0;
}

//______________________________________________________________________________
TBranch *TProofChain::Bronch(const char *, const char *, void *, Int_t, Int_t)
{
   // Not implemented in TProofChain. Shouldn't be used.

   Error("TProofChain::Bronch","not implemented");
   return 0;
}

//______________________________________________________________________________
void TProofChain::Browse(TBrowser *b)
{
   // Forwards the execution to the dummy tree header.
   // See TTree::Browse().

   fSet->Browse(b);
}

//______________________________________________________________________________
Int_t TProofChain::BuildIndex(const char *, const char *)
{
   // Not implemented in TProofChain. Shouldn't be used.

   Error("TProofChain::BuildIndex","not implemented");
   return 0;
}

//______________________________________________________________________________
void TProofChain::SetTreeIndex(TVirtualIndex *)
{
   // Not implemented in TProofChain. Shouldn't be used.

   Error("TProofChain::SetTreeIndex","not implemented");
}

//______________________________________________________________________________
TStreamerInfo *TProofChain::BuildStreamerInfo(TClass *, void *)
{
   // Not implemented in TProofChain. Shouldn't be used.

   Error("TProofChain::BuildStreamerInfo","not implemented");
   return 0;
}

//______________________________________________________________________________
TFile *TProofChain::ChangeFile(TFile *)
{
   // Not implemented in TProofChain. Shouldn't be used.

   Error("TProofChain::ChangeFile","not implemented");
   return 0;
}

//______________________________________________________________________________
TTree *TProofChain::CloneTree(Long64_t, Option_t *)
{
   // Not implemented in TProofChain. Shouldn't be used.

   Error("TProofChain::CloneTree","not implemented");
   return 0;
}

//______________________________________________________________________________
void TProofChain::CopyAddresses(TTree* )
{
   // Not implemented in TProofChain. Shouldn't be used.

   Error("TProofChain::CopyAddresses","not implemented");
}

//______________________________________________________________________________
Long64_t TProofChain::CopyEntries(TTree *, Long64_t)
{
   // Not implemented in TProofChain. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
TTree *TProofChain::CopyTree(const char *, Option_t *, Long64_t , Long64_t )
{
   // Not implemented in TProofChain. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
void TProofChain::Delete(Option_t *)
{
   // Not implemented in TProofChain. Shouldn't be used.

   return;
}

//______________________________________________________________________________
Long64_t TProofChain::Draw(const char *varexp, const TCut &selection, Option_t *option, Long64_t nentries, Long64_t firstentry)
{
   // Forwards the execution to the TDSet.
   // See TDSet::Browse().

   if (!fProof && gProof) {
      ConnectProof(gProof);
   }
   fProof->SetDrawFeedbackOption(fDrawFeedback, option);
   fReadEntry = firstentry;
   fSet->SetEventList(fEventList);

   Long64_t rv = fSet->Draw(varexp, selection, option, nentries, firstentry);
   return rv;
}

//______________________________________________________________________________
Long64_t TProofChain::Draw(const char *varexp, const char *selection, Option_t *option,Long64_t nentries, Long64_t firstentry)
{
   // Forwards the execution to the TDSet.
   // See TDSet::Browse().

   if (!fProof && gProof) {
      ConnectProof(gProof);
   }
   fProof->SetDrawFeedbackOption(fDrawFeedback, option);
   fReadEntry = firstentry;
   fSet->SetEventList(fEventList);
   Long64_t rv = fSet->Draw(varexp, selection, option, nentries, firstentry);
   return rv;
}

//______________________________________________________________________________
void TProofChain::DropBuffers(Int_t)
{
   // Not implemented in TProofChain. Shouldn't be used.

   return;
}

//______________________________________________________________________________
Int_t TProofChain::Fill()
{
   // Not implemented in TProofChain. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
TBranch *TProofChain::FindBranch(const char* branchname)
{
   // Forwards the execution to the dummy tree header.
   // See TTree::FindBranch().

   return fTree->FindBranch(branchname);
}

//______________________________________________________________________________
TLeaf *TProofChain::FindLeaf(const char* searchname)
{
   // Forwards the execution to the dummy tree header.
   // See TTree::FindLeaf().

   return fTree->FindLeaf(searchname);
}

//______________________________________________________________________________
Long64_t TProofChain::Fit(const char * ,const char *, const char *, Option_t *,
                          Option_t *, Long64_t, Long64_t)
{
   // Not implemented in TProofChain. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
const char *TProofChain::GetAlias(const char *) const
{
   // Not implemented in TProofChain. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
TBranch *TProofChain::GetBranch(const char *name)
{
   // Forwards the execution to the dummy tree header.
   // See TTree::GetBranch().

   return fTree->GetBranch(name);
}

//______________________________________________________________________________
Bool_t TProofChain::GetBranchStatus(const char *branchname) const
{
   // Forwards the execution to the dummy tree header.
   // See TTree::GetBranchStatus().

   return fTree->GetBranchStatus(branchname);
}

//______________________________________________________________________________
Int_t TProofChain::GetBranchStyle()
{
   // See TTree::GetBranchStyle().

   return fgBranchStyle;
}

//______________________________________________________________________________
TFile *TProofChain::GetCurrentFile() const
{
   // Not implemented in TProofChain. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
Long64_t TProofChain::GetEntriesFriend() const
{
   // Not implemented in TProofChain. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
Int_t TProofChain::GetEntry(Long64_t, Int_t)
{
   // Not implemented in TProofChain. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
Long64_t TProofChain::GetEntryNumber(Long64_t) const
{
   // Not implemented in TProofChain. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
Long64_t TProofChain::GetEntryNumberWithBestIndex(Int_t, Int_t) const
{
   // Not implemented in TProofChain. Shouldn't be used.

   return 0;
}


//______________________________________________________________________________
Long64_t TProofChain::GetEntryNumberWithIndex(Int_t, Int_t) const
{
   // Not implemented in TProofChain. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
Int_t TProofChain::GetEntryWithIndex(Int_t, Int_t)
{
   // Not implemented in TProofChain. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
const char *TProofChain::GetFriendAlias(TTree *) const
{
   // Not implemented in TProofChain. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
TIterator* TProofChain::GetIteratorOnAllLeaves(Bool_t)
{
   // Not implemented in TProofChain. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
TLeaf *TProofChain::GetLeaf(const char *)
{
   // Not implemented in TProofChain. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
Double_t TProofChain::GetMaximum(const char *)
{
   // Not implemented in TProofChain. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
Long64_t TProofChain::GetMaxTreeSize()
{
   // Not implemented in TProofChain. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
Double_t TProofChain::GetMinimum(const char *)
{
   // Not implemented in TProofChain. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
const char *TProofChain::GetNameByIndex(TString &, Int_t *, Int_t) const
{
   // Not implemented in TProofChain. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
TVirtualTreePlayer *TProofChain::GetPlayer()
{
   // Forwards the execution to the dummy tree header.
   // See TTree::GetPlayer().

   return fTree->GetPlayer();    // FIXME ??
}

//______________________________________________________________________________
TList *TProofChain::GetUserInfo()
{
   // Not implemented in TProofChain. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
void TProofChain::KeepCircular()
{
   // Not implemented in TProofChain. Shouldn't be used.

   return;
}

//______________________________________________________________________________
Long64_t TProofChain::LoadTree(Long64_t)
{
   // Not implemented in TProofChain. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
Int_t TProofChain::LoadBaskets(Long64_t)
{
   // Not implemented in TProofChain. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
Long64_t TProofChain::LoadTreeFriend(Long64_t, TTree *)
{
   // Not implemented in TProofChain. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
Int_t TProofChain::MakeSelector(const char *)
{
   // Not implemented in TProofChain. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
Int_t TProofChain::MakeProxy(const char *, const char *,
                             const char *, const char *,
                             Int_t)
{
   // Not implemented in TProofChain. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
Int_t TProofChain::MakeClass(const char *, Option_t *)
{
   // Not implemented in TProofChain. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
Int_t TProofChain::MakeCode(const char *)
{
   // Not implemented in TProofChain. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
void TProofChain::MakeIndex(TString &, Int_t *)
{
   // Not implemented in TProofChain. Shouldn't be used.

   return;
}

//______________________________________________________________________________
Bool_t TProofChain::MemoryFull(Int_t)
{
   // Not implemented in TProofChain. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
TTree *TProofChain::MergeTrees(TList *)
{
   // Not implemented in TProofChain. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
Long64_t TProofChain::Merge(TCollection *)
{
   // Not implemented in TProofChain. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
Bool_t TProofChain::Notify()
{
   // Not implemented in TProofChain. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
TPrincipal *TProofChain::Principal(const char *, const char *, Option_t *,
                                   Long64_t, Long64_t)
{
   // Not implemented in TProofChain. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
void TProofChain::Print(Option_t *) const
{
   // Not implemented in TProofChain. Shouldn't be used.

   return;
}

//______________________________________________________________________________
Long64_t TProofChain::Process(const char *filename,Option_t *option,Long64_t nentries, Long64_t firstentry)
{
   // Forwards the execution to the TDSet.
   // See TDSet::Process().

   return fSet->Process(filename, option, nentries, firstentry);
}

//______________________________________________________________________________
Long64_t TProofChain::Process(TSelector *selector,Option_t *option, Long64_t nentries, Long64_t firstentry)
{
   // Not implemented in TProofChain. Shouldn't be used.

   if (selector ||  option || nentries || firstentry);
   //   return fSet->Process(selector, option, nentries, firstentry);
   Warning("Process", "not implemented"); // TODO
   return 0;
}

//______________________________________________________________________________
Long64_t TProofChain::Project(const char *, const char *, const char *,
                              Option_t *, Long64_t, Long64_t)
{
   // Not implemented in TProofChain. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
TSQLResult *TProofChain::Query(const char *, const char *, Option_t *,
                               Long64_t, Long64_t)
{
   // Not implemented in TProofChain. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
Long64_t TProofChain::ReadFile(const char *, const char *)
{
   // Not implemented in TProofChain. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
void TProofChain::Refresh()
{
   // Not implemented in TProofChain. Shouldn't be used.
}

//______________________________________________________________________________
void TProofChain::RemoveFriend(TTree *)
{
   // Not implemented in TProofChain. Shouldn't be used.
}

//______________________________________________________________________________
void TProofChain::Reset(Option_t *)
{
   // Not implemented in TProofChain. Shouldn't be used.
}

//______________________________________________________________________________
void TProofChain::ResetBranchAddresses()
{
   // Not implemented in TProofChain. Shouldn't be used.
}

//______________________________________________________________________________
Long64_t  TProofChain::Scan(const char *, const char *, Option_t *,
                            Long64_t, Long64_t)
{
   // Not implemented in TProofChain. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
Bool_t TProofChain::SetAlias(const char *, const char *)
{
   // Not implemented in TProofChain. Shouldn't be used.

   return 0;
}

//_______________________________________________________________________
void TProofChain::SetBasketSize(const char *, Int_t)
{
   // Not implemented in TProofChain. Shouldn't be used.
}

//_______________________________________________________________________
void TProofChain::SetBranchAddress(const char *, void *)
{
   // Not implemented in TProofChain. Shouldn't be used.
}
//_______________________________________________________________________
void TProofChain::SetBranchAddress(const char *, void *,
                                   TClass *, EDataType, Bool_t)
{
   // Not implemented in TProofChain. Shouldn't be used.
}

//_______________________________________________________________________
void TProofChain::SetBranchStatus(const char *, Bool_t, UInt_t *)
{
   // Not implemented in TProofChain. Shouldn't be used.
}


//______________________________________________________________________________
void TProofChain::SetBranchStyle(Int_t)
{
   // Not implemented in TProofChain. Shouldn't be used.
}

//______________________________________________________________________________
void TProofChain::SetCircular(Long64_t)
{
   // Not implemented in TProofChain. Shouldn't be used.
}

//______________________________________________________________________________
void TProofChain::SetDebug(Int_t level, Long64_t min, Long64_t max)
{
   // See TTree::SetDebug

   TTree::SetDebug(level, min, max);
}

//______________________________________________________________________________
void TProofChain::SetDirectory(TDirectory *)
{
   // Not implemented in TProofChain. Shouldn't be used.
}

//_______________________________________________________________________
Long64_t TProofChain::SetEntries(Long64_t)
{
   // Not implemented in TProofChain. Shouldn't be used.

   return 0;
}

//_______________________________________________________________________
void TProofChain::SetEstimate(Long64_t)
{
   // Not implemented in TProofChain. Shouldn't be used.
}

//_______________________________________________________________________
void TProofChain::SetFileNumber(Int_t)
{
   // Not implemented in TProofChain. Shouldn't be used.
}

//______________________________________________________________________________
void TProofChain::SetMaxTreeSize(Long64_t)
{
   // Not implemented in TProofChain. Shouldn't be used.
}

//______________________________________________________________________________
void TProofChain::SetName(const char *name)
{
   // See TTree::GetName.

   TTree::SetName(name);
}

//______________________________________________________________________________
void TProofChain::SetObject(const char *, const char *)
{
   // Not implemented in TProofChain. Shouldn't be used.
}

//______________________________________________________________________________
void TProofChain::SetWeight(Double_t, Option_t *)
{
   // Not implemented in TProofChain. Shouldn't be used.
}

//_______________________________________________________________________
void TProofChain::Show(Long64_t, Int_t)
{
   // Not implemented in TProofChain. Shouldn't be used.
}

//_______________________________________________________________________
void TProofChain::StartViewer()
{
   // Not implemented in TProofChain. Shouldn't be used.
}

//_______________________________________________________________________
void TProofChain::Streamer(TBuffer &)
{
   // Not implemented in TProofChain. Shouldn't be used.
}

//______________________________________________________________________________
Long64_t TProofChain::UnbinnedFit(const char * ,const char *, const char *,
                                  Option_t *,Long64_t, Long64_t)
{
   // Not implemented in TProofChain. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
void TProofChain::UseCurrentStyle()
{
   // Not implemented in TProofChain. Shouldn't be used.
}

//______________________________________________________________________________
Long64_t TProofChain::GetEntries() const
{
   // Returns the total number of entries in the TProofChain, which is
   // the number of entries in the TDSet that it holds.

   return fTree->GetMaxEntryLoop();  // this was used for holding the total number of entries
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

   if (!fProof)
      return;
   fProof->Disconnect("Progress(Long64_t,Long64_t)",
                      this, "Progress(Long64_t,Long64_t)");
   fProof->DeleteDrawFeedback(fDrawFeedback);
   fDrawFeedback = 0;
   fProof = 0;

}
//______________________________________________________________________________
void TProofChain::ConnectProof(TVirtualProof *proof)
{
   // Connects the proof - creates a "DrawFeedback" and connects the
   // "Progress" signal.

   if (fProof)
      ReleaseProof();
   fProof = proof;

   fDrawFeedback = fProof->CreateDrawFeedback();

   fProof->Connect("Progress(Long64_t,Long64_t)", "TProofChain",
                   this, "Progress(Long64_t,Long64_t)");
}

//______________________________________________________________________________
TProofChain *TProofChain::MakeProofChain(TDSet *set, TVirtualProof *proof)
{
   // Creates a new TProofChain that keeps the TDSet.
   // uses proof to get the three header.

   if (!set->IsTree()) {
      set->Error("MakeProofChain", "TDSet contents should be of type TTree (or subtype)");
      return 0;
   }

   TTree *t = proof->GetTreeHeader(set);
   if (!t) {
      set->Error("TProofChain::MakeProofChain", "Error getting a tree header");
      return 0;
   }
   TProofChain *w = new TProofChain(set, t);   // t will be deleted in w's destructor
   w->ConnectProof(proof);
   w->SetDirectory(0);
   w->SetName(TString(t->GetName())  + "_Wrapped");
   return w;
}
