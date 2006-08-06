// @(#)root/tree:$Name:  $:$Id: TChainProof.cxx,v 1.7 2006/07/05 17:24:57 brun Exp $
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
// TChainProof                                                          //
//                                                                      //
// A wrapper for TDSet to behave as a Tree/Chain.                       //
// Uses an internal TDSet to handle processing and a TTree              //
// which holds the branch structure.                                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TChainProof.h"
#include "TDSet.h"
#include "TVirtualProof.h"


ClassImp(TChainProof)

//______________________________________________________________________________
TChainProof::TChainProof(TDSet *set, TTree *tree, TVirtualProof* proof) : TTree()
{
   // Crates a new TProof chain containing the files from the TDSet.
   // The tree is just a dummy containing descriptions of all the tree leaves.

   fTree         = tree;
   fSet          = set;
   fDirectory    = gDirectory;
   fProof        = proof;
   fDrawFeedback = 0;
}

//______________________________________________________________________________
TChainProof::~TChainProof()
{
   // Destructor - removes the chain from the proof in case a proof was set.

   ReleaseProof();
   SafeDelete(fTree);
   fDirectory    = 0;
}

//______________________________________________________________________________
void TChainProof::AddClone(TTree *)
{
   // Not implemented in TChainProof. Shouldn't be used.

   Error("TChainProof::AddClone","not implemented");
}

//______________________________________________________________________________
TFriendElement *TChainProof::AddFriend(const char *, const char *)
{
   // Not implemented in TChainProof. Shouldn't be used.

   return 0;
}
//______________________________________________________________________________
TFriendElement *TChainProof::AddFriend(const char *, TFile *)
{
   // Not implemented in TChainProof. Shouldn't be used.

   return 0;

}
//______________________________________________________________________________
TFriendElement *TChainProof::AddFriend(TTree *, const char*, Bool_t)
{
   // Not implemented in TChainProof. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
Long64_t TChainProof::AutoSave(Option_t *)
{
   // Not implemented in TChainProof. Shouldn't be used.

   Error("TChainProof::AutoSave","not implemented");
   return 0;
}

//______________________________________________________________________________
Int_t TChainProof::Branch(TList *, Int_t, Int_t)
{
   // Not implemented in TChainProof. Shouldn't be used.

   Error("TChainProof::Branch","not implemented");
   return 0;
}

//______________________________________________________________________________
Int_t TChainProof::Branch(TCollection *, Int_t, Int_t, const char *)
{
   // Not implemented in TChainProof. Shouldn't be used.

   Error("TChainProof::Branch","not implemented");
   return 0;
}

//______________________________________________________________________________
Int_t TChainProof::Branch(const char *, Int_t, Int_t)
{
   // Not implemented in TChainProof. Shouldn't be used.

   Error("TChainProof::Branch","not implemented");
   return 0;
}

//______________________________________________________________________________
TBranch *TChainProof::Branch(const char *, TClonesArray **, Int_t, Int_t)
{
   // Not implemented in TChainProof. Shouldn't be used.

   Error("TChainProof::Branch","not implemented");
   return 0;
}

//______________________________________________________________________________
TBranch *TChainProof::Branch(const char *, void *, const char *, Int_t)
{
   // Not implemented in TChainProof. Shouldn't be used.

   Error("TChainProof::Branch","not implemented");
   return 0;
}

//______________________________________________________________________________
TBranch *TChainProof::Branch(const char *, void *, Int_t, Int_t)
{
   // Not implemented in TChainProof. Shouldn't be used.

   Error("TChainProof::Branch","not implemented");
   return 0;
}

//______________________________________________________________________________
TBranch *TChainProof::Branch(const char *, const char *, void *, Int_t, Int_t)
{
   // Not implemented in TChainProof. Shouldn't be used.

   Error("TChainProof::Branch","not implemented");
   return 0;
}

//______________________________________________________________________________
TBranch *TChainProof::BranchOld(const char *, const char *, void *, Int_t, Int_t)
{
   // Not implemented in TChainProof. Shouldn't be used.

   Error("TChainProof::BranchOld","not implemented");
   return 0;
}

//______________________________________________________________________________
TBranch *TChainProof::BranchRef()
{
   // Not implemented in TChainProof. Shouldn't be used.

   Error("TChainProof::BranchRef", "not implemented");
   return 0;
}

//______________________________________________________________________________
TBranch *TChainProof::Bronch(const char *, const char *, void *, Int_t, Int_t)
{
   // Not implemented in TChainProof. Shouldn't be used.

   Error("TChainProof::Bronch","not implemented");
   return 0;
}

//______________________________________________________________________________
void TChainProof::Browse(TBrowser *b)
{
   // Forwards the execution to the dummy tree header.
   // See TTree::Browse().

   fSet->Browse(b);
}

//______________________________________________________________________________
Int_t TChainProof::BuildIndex(const char *, const char *)
{
   // Not implemented in TChainProof. Shouldn't be used.

   Error("TChainProof::BuildIndex","not implemented");
   return 0;
}

//______________________________________________________________________________
void TChainProof::SetTreeIndex(TVirtualIndex *)
{
   // Not implemented in TChainProof. Shouldn't be used.

   Error("TChainProof::SetTreeIndex","not implemented");
}

//______________________________________________________________________________
TStreamerInfo *TChainProof::BuildStreamerInfo(TClass *, void *)
{
   // Not implemented in TChainProof. Shouldn't be used.

   Error("TChainProof::BuildStreamerInfo","not implemented");
   return 0;
}

//______________________________________________________________________________
TFile *TChainProof::ChangeFile(TFile *)
{
   // Not implemented in TChainProof. Shouldn't be used.

   Error("TChainProof::ChangeFile","not implemented");
   return 0;
}

//______________________________________________________________________________
TTree *TChainProof::CloneTree(Long64_t, Option_t *)
{
   // Not implemented in TChainProof. Shouldn't be used.

   Error("TChainProof::CloneTree","not implemented");
   return 0;
}

//______________________________________________________________________________
void TChainProof::CopyAddresses(TTree* )
{
   // Not implemented in TChainProof. Shouldn't be used.

   Error("TChainProof::CopyAddresses","not implemented");
}

//______________________________________________________________________________
Long64_t TChainProof::CopyEntries(TTree *, Long64_t)
{
   // Not implemented in TChainProof. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
TTree *TChainProof::CopyTree(const char *, Option_t *, Long64_t , Long64_t )
{
   // Not implemented in TChainProof. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
void TChainProof::Delete(Option_t *)
{
   // Not implemented in TChainProof. Shouldn't be used.

   return;
}

//______________________________________________________________________________
Long64_t TChainProof::Draw(const char *varexp, const TCut &selection, Option_t *option, Long64_t nentries, Long64_t firstentry)
{
   // Forwards the execution to the TDSet.
   // Returns -1 in case of error or number of selected events in case of success.
   // See TDSet::Browse().

   if (!fProof && gProof) {
      ConnectProof(gProof);
   }
   if (fDrawFeedback)
      fProof->SetDrawFeedbackOption(fDrawFeedback, option);
   fReadEntry = firstentry;
   fSet->SetEventList(fEventList);

   Long64_t rv = fSet->Draw(varexp, selection, option, nentries, firstentry);
   return rv;
}

//______________________________________________________________________________
Long64_t TChainProof::Draw(const char *varexp, const char *selection, Option_t *option,Long64_t nentries, Long64_t firstentry)
{
   // Forwards the execution to the TDSet.
   // Returns -1 in case of error or number of selected events in case of success.
   // See TDSet::Browse().

   if (!fProof && gProof) {
      ConnectProof(gProof);
   }
   if (fDrawFeedback)
      fProof->SetDrawFeedbackOption(fDrawFeedback, option);
   fReadEntry = firstentry;
   fSet->SetEventList(fEventList);
   Long64_t rv = fSet->Draw(varexp, selection, option, nentries, firstentry);
   return rv;
}

//______________________________________________________________________________
void TChainProof::DropBuffers(Int_t)
{
   // Not implemented in TChainProof. Shouldn't be used.

   return;
}

//______________________________________________________________________________
Int_t TChainProof::Fill()
{
   // Not implemented in TChainProof. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
TBranch *TChainProof::FindBranch(const char* branchname)
{
   // Forwards the execution to the dummy tree header.
   // See TTree::FindBranch().

   return (fTree ? fTree->FindBranch(branchname) : (TBranch *)0);
}

//______________________________________________________________________________
TLeaf *TChainProof::FindLeaf(const char* searchname)
{
   // Forwards the execution to the dummy tree header.
   // See TTree::FindLeaf().

   return (fTree ? fTree->FindLeaf(searchname) : (TLeaf *)0);
}

//______________________________________________________________________________
Long64_t TChainProof::Fit(const char * ,const char *, const char *, Option_t *,
                          Option_t *, Long64_t, Long64_t)
{
   // Not implemented in TChainProof. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
const char *TChainProof::GetAlias(const char *) const
{
   // Not implemented in TChainProof. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
TBranch *TChainProof::GetBranch(const char *name)
{
   // Forwards the execution to the dummy tree header.
   // See TTree::GetBranch().

   return (fTree ? fTree->GetBranch(name) : (TBranch *)0);
}

//______________________________________________________________________________
Bool_t TChainProof::GetBranchStatus(const char *branchname) const
{
   // Forwards the execution to the dummy tree header.
   // See TTree::GetBranchStatus().

   return (fTree ? fTree->GetBranchStatus(branchname) : kFALSE);
}

//______________________________________________________________________________
Int_t TChainProof::GetBranchStyle()
{
   // See TTree::GetBranchStyle().

   return fgBranchStyle;
}

//______________________________________________________________________________
TFile *TChainProof::GetCurrentFile() const
{
   // Not implemented in TChainProof. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
Long64_t TChainProof::GetEntriesFriend() const
{
   // Not implemented in TChainProof. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
Int_t TChainProof::GetEntry(Long64_t, Int_t)
{
   // Not implemented in TChainProof. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
Long64_t TChainProof::GetEntryNumber(Long64_t) const
{
   // Not implemented in TChainProof. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
Long64_t TChainProof::GetEntryNumberWithBestIndex(Int_t, Int_t) const
{
   // Not implemented in TChainProof. Shouldn't be used.

   return 0;
}


//______________________________________________________________________________
Long64_t TChainProof::GetEntryNumberWithIndex(Int_t, Int_t) const
{
   // Not implemented in TChainProof. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
Int_t TChainProof::GetEntryWithIndex(Int_t, Int_t)
{
   // Not implemented in TChainProof. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
const char *TChainProof::GetFriendAlias(TTree *) const
{
   // Not implemented in TChainProof. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
TIterator* TChainProof::GetIteratorOnAllLeaves(Bool_t)
{
   // Not implemented in TChainProof. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
TLeaf *TChainProof::GetLeaf(const char *)
{
   // Not implemented in TChainProof. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
Double_t TChainProof::GetMaximum(const char *)
{
   // Not implemented in TChainProof. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
Long64_t TChainProof::GetMaxTreeSize()
{
   // Not implemented in TChainProof. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
Double_t TChainProof::GetMinimum(const char *)
{
   // Not implemented in TChainProof. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
const char *TChainProof::GetNameByIndex(TString &, Int_t *, Int_t) const
{
   // Not implemented in TChainProof. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
TVirtualTreePlayer *TChainProof::GetPlayer()
{
   // Forwards the execution to the dummy tree header.
   // See TTree::GetPlayer().

   if (!fTree) {
      if (fProof) {
         fTree = fProof->GetTreeHeader(fSet);
         ConnectProof(fProof);
      }
   }

   return (fTree ? fTree->GetPlayer() : (TVirtualTreePlayer *)0);    // FIXME ??
}

//______________________________________________________________________________
TList *TChainProof::GetUserInfo()
{
   // Not implemented in TChainProof. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
void TChainProof::KeepCircular()
{
   // Not implemented in TChainProof. Shouldn't be used.

   return;
}

//______________________________________________________________________________
Long64_t TChainProof::LoadTree(Long64_t)
{
   // Not implemented in TChainProof. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
Int_t TChainProof::LoadBaskets(Long64_t)
{
   // Not implemented in TChainProof. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
Long64_t TChainProof::LoadTreeFriend(Long64_t, TTree *)
{
   // Not implemented in TChainProof. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
Int_t TChainProof::MakeSelector(const char *)
{
   // Not implemented in TChainProof. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
Int_t TChainProof::MakeProxy(const char *, const char *,
                             const char *, const char *,
                             Int_t)
{
   // Not implemented in TChainProof. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
Int_t TChainProof::MakeClass(const char *, Option_t *)
{
   // Not implemented in TChainProof. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
Int_t TChainProof::MakeCode(const char *)
{
   // Not implemented in TChainProof. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
void TChainProof::MakeIndex(TString &, Int_t *)
{
   // Not implemented in TChainProof. Shouldn't be used.

   return;
}

//______________________________________________________________________________
Bool_t TChainProof::MemoryFull(Int_t)
{
   // Not implemented in TChainProof. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
TTree *TChainProof::MergeTrees(TList *,Option_t *)
{
   // Not implemented in TChainProof. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
Long64_t TChainProof::Merge(TCollection *,Option_t *)
{
   // Not implemented in TChainProof. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
Bool_t TChainProof::Notify()
{
   // Not implemented in TChainProof. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
TPrincipal *TChainProof::Principal(const char *, const char *, Option_t *,
                                   Long64_t, Long64_t)
{
   // Not implemented in TChainProof. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
void TChainProof::Print(Option_t *) const
{
   // Not implemented in TChainProof. Shouldn't be used.

   return;
}

//______________________________________________________________________________
Long64_t TChainProof::Process(const char *filename,Option_t *option,Long64_t nentries, Long64_t firstentry)
{
   // Forwards the execution to the TDSet.
   // The return value is -1 in case of error and TSelector::GetStatus() in
   // in case of success.
   // See TDSet::Process().

   return fSet->Process(filename, option, nentries, firstentry);
}

//______________________________________________________________________________
Long64_t TChainProof::Process(TSelector *selector,Option_t *option, Long64_t nentries, Long64_t firstentry)
{
   // Not implemented in TChainProof. Shouldn't be used.
   // The return value is -1 in case of error and TSelector::GetStatus() in
   // in case of success.

   if (selector ||  option || nentries || firstentry);
   //   return fSet->Process(selector, option, nentries, firstentry);
   Warning("Process", "not implemented"); // TODO
   return -1;
}

//______________________________________________________________________________
Long64_t TChainProof::Project(const char *, const char *, const char *,
                              Option_t *, Long64_t, Long64_t)
{
   // Not implemented in TChainProof. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
TSQLResult *TChainProof::Query(const char *, const char *, Option_t *,
                               Long64_t, Long64_t)
{
   // Not implemented in TChainProof. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
Long64_t TChainProof::ReadFile(const char *, const char *)
{
   // Not implemented in TChainProof. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
void TChainProof::Refresh()
{
   // Not implemented in TChainProof. Shouldn't be used.
}

//______________________________________________________________________________
void TChainProof::RemoveFriend(TTree *)
{
   // Not implemented in TChainProof. Shouldn't be used.
}

//______________________________________________________________________________
void TChainProof::Reset(Option_t *)
{
   // Not implemented in TChainProof. Shouldn't be used.
}

//______________________________________________________________________________
void TChainProof::ResetBranchAddresses()
{
   // Not implemented in TChainProof. Shouldn't be used.
}

//______________________________________________________________________________
Long64_t  TChainProof::Scan(const char *, const char *, Option_t *,
                            Long64_t, Long64_t)
{
   // Not implemented in TChainProof. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
Bool_t TChainProof::SetAlias(const char *, const char *)
{
   // Not implemented in TChainProof. Shouldn't be used.

   return 0;
}

//_______________________________________________________________________
void TChainProof::SetBasketSize(const char *, Int_t)
{
   // Not implemented in TChainProof. Shouldn't be used.
}

//_______________________________________________________________________
void TChainProof::SetBranchAddress(const char *, void *, TBranch **)
{
   // Not implemented in TChainProof. Shouldn't be used.
}
//_______________________________________________________________________
void TChainProof::SetBranchAddress(const char *, void *,
                                   TClass *, EDataType, Bool_t)
{
   // Not implemented in TChainProof. Shouldn't be used.
}

//_______________________________________________________________________
void TChainProof::SetBranchAddress(const char *, void *, TBranch **,
                                   TClass *, EDataType, Bool_t)
{
   // Not implemented in TChainProof. Shouldn't be used.
}

//_______________________________________________________________________
void TChainProof::SetBranchStatus(const char *, Bool_t, UInt_t *)
{
   // Not implemented in TChainProof. Shouldn't be used.
}


//______________________________________________________________________________
void TChainProof::SetBranchStyle(Int_t)
{
   // Not implemented in TChainProof. Shouldn't be used.
}

//______________________________________________________________________________
void TChainProof::SetCircular(Long64_t)
{
   // Not implemented in TChainProof. Shouldn't be used.
}

//______________________________________________________________________________
void TChainProof::SetDebug(Int_t level, Long64_t min, Long64_t max)
{
   // See TTree::SetDebug

   TTree::SetDebug(level, min, max);
}

//______________________________________________________________________________
void TChainProof::SetDirectory(TDirectory *)
{
   // Not implemented in TChainProof. Shouldn't be used.
}

//_______________________________________________________________________
Long64_t TChainProof::SetEntries(Long64_t)
{
   // Not implemented in TChainProof. Shouldn't be used.

   return 0;
}

//_______________________________________________________________________
void TChainProof::SetEstimate(Long64_t)
{
   // Not implemented in TChainProof. Shouldn't be used.
}

//_______________________________________________________________________
void TChainProof::SetFileNumber(Int_t)
{
   // Not implemented in TChainProof. Shouldn't be used.
}

//______________________________________________________________________________
void TChainProof::SetMaxTreeSize(Long64_t)
{
   // Not implemented in TChainProof. Shouldn't be used.
}

//______________________________________________________________________________
void TChainProof::SetName(const char *name)
{
   // See TTree::GetName.

   TTree::SetName(name);
}

//______________________________________________________________________________
void TChainProof::SetObject(const char *, const char *)
{
   // Not implemented in TChainProof. Shouldn't be used.
}

//______________________________________________________________________________
void TChainProof::SetWeight(Double_t, Option_t *)
{
   // Not implemented in TChainProof. Shouldn't be used.
}

//_______________________________________________________________________
void TChainProof::Show(Long64_t, Int_t)
{
   // Not implemented in TChainProof. Shouldn't be used.
}

//_______________________________________________________________________
void TChainProof::StartViewer()
{
   // Not implemented in TChainProof. Shouldn't be used.
}

//_______________________________________________________________________
void TChainProof::Streamer(TBuffer &)
{
   // Not implemented in TChainProof. Shouldn't be used.
}

//______________________________________________________________________________
Long64_t TChainProof::UnbinnedFit(const char * ,const char *, const char *,
                                  Option_t *,Long64_t, Long64_t)
{
   // Not implemented in TChainProof. Shouldn't be used.

   return 0;
}

//______________________________________________________________________________
void TChainProof::UseCurrentStyle()
{
   // Not implemented in TChainProof. Shouldn't be used.
}

//______________________________________________________________________________
Long64_t TChainProof::GetEntries() const
{
   // Returns the total number of entries in the TChainProof, which is
   // the number of entries in the TDSet that it holds.

   return (fTree ? fTree->GetMaxEntryLoop() : (Long64_t)(-1));  // this was used for holding the total number of entries
}

//______________________________________________________________________________
void TChainProof::Progress(Long64_t total, Long64_t processed)
{
   // Changes the number of processed entries.

   if (gROOT->IsInterrupted() && gProof)
      gProof->StopProcess(kTRUE);
   if (total)
      ;

   fReadEntry = processed;
}

//______________________________________________________________________________
Long64_t TChainProof::GetReadEntry() const
{
   // Returns the number of processed entries.

   return fReadEntry;
}

//______________________________________________________________________________
void TChainProof::ReleaseProof()
{
   // Releases PROOF. Disconnect the "progress" signal.

   if (!fProof)
      return;
   fProof->Disconnect("Progress(Long64_t,Long64_t)",
                      this, "Progress(Long64_t,Long64_t)");
   if (fDrawFeedback)
      fProof->DeleteDrawFeedback(fDrawFeedback);
   fDrawFeedback = 0;
   fProof = 0;
}

//______________________________________________________________________________
void TChainProof::ConnectProof(TVirtualProof *proof)
{
   // Connects the proof - creates a "DrawFeedback" and connects the
   // "Progress" signal.

   if (fProof)
      ReleaseProof();
   fProof = proof;

   if (fProof) {
      fDrawFeedback = fProof->CreateDrawFeedback();

      fProof->Connect("Progress(Long64_t,Long64_t)", "TChainProof",
                       this, "Progress(Long64_t,Long64_t)");
   }
}

//______________________________________________________________________________
TChainProof *TChainProof::MakeChainProof(TDSet *set, TVirtualProof *proof, Bool_t gettreeheader)
{
   // Creates a new TChainProof that keeps the TDSet.
   // If gettreeheader is kTRUE the header of the tree will be read from the
   // PROOF cluster: this is only needed fro browsing and should be used with
   // care because it may take a long time to execute.

   if (!set->IsTree()) {
      set->Error("MakeChainProof", "TDSet contents should be of type TTree (or subtype)");
      return 0;
   }

   TTree *t = 0;
   if (gettreeheader) {
      if (!(t = proof->GetTreeHeader(set))) {
         set->Error("TChainProof::MakeChainProof", "Error getting a tree header");
         return 0;
      }
   }
   TChainProof *w = new TChainProof(set, t, proof);   // t will be deleted in w's destructor
   w->SetDirectory(0);
   if (t)
      w->SetName(TString(t->GetName())  + "_Wrapped");
   return w;
}
