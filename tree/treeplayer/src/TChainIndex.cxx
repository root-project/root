// @(#)root/tree:$Id$
// Author: Marek Biskup   07/06/2005

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//
// A Chain Index
//
//////////////////////////////////////////////////////////////////////////

#include "TChainIndex.h"
#include "TChain.h"
#include "TTreeFormula.h"
#include "TTreeIndex.h"
#include "TFile.h"
#include "TError.h"

ClassImp(TChainIndex)

//______________________________________________________________________________
TChainIndex::TChainIndex(): TVirtualIndex()
{
// Default constructor for TChainIndex

   fTree = 0;
   fMajorFormulaParent = fMinorFormulaParent = 0;
}

//______________________________________________________________________________
TChainIndex::TChainIndex(const TTree *T, const char *majorname, const char *minorname)
           : TVirtualIndex()
{
   // Normal constructor for TChainIndex. See TTreeIndex::TTreeIndex for the description of the
   // parameters.
   // The tree must be a TChain.
   // All the index values in the first tree of the chain must be
   // less then any index value in the second one, and so on.
   // If any of those requirements isn't met the object becomes a zombie.
   // If some subtrees don't have indices the indices are created and stored inside this
   // TChainIndex.

   fTree = 0;
   fMajorFormulaParent = fMinorFormulaParent = 0;

   TChain *chain = dynamic_cast<TChain*>(const_cast<TTree*>(T));
   if (!chain) {
      MakeZombie();
      Error("TChainIndex", "Cannot create a TChainIndex."
            " The Tree passed as an argument is not a TChain");
      return;
   }

   fTree               = (TTree*)T;
   fMajorName          = majorname;
   fMinorName          = minorname;
   Int_t i = 0;

   // Go through all the trees and check if they have indeces. If not then build them.
   for (i = 0; i < chain->GetNtrees(); i++) {
      chain->LoadTree((chain->GetTreeOffset())[i]);
      TVirtualIndex *index = chain->GetTree()->GetTreeIndex();

      TChainIndexEntry entry;
      entry.fTreeIndex = 0;

      //if an index already exists, we must check if major/minorname correspond
      //to the major/minor names in this function call
      if (index) {
         if (strcmp(majorname,index->GetMajorName()) || strcmp(minorname,index->GetMinorName())) {
            MakeZombie();
            Error("TChainIndex","Tree in file %s has an index built with majorname=%s and minorname=%s",chain->GetTree()->GetCurrentFile()->GetName(),index->GetMajorName(),index->GetMinorName());
            return;
         }
      }
      if (!index) {
         chain->GetTree()->BuildIndex(majorname, minorname);
         index = chain->GetTree()->GetTreeIndex();
         chain->GetTree()->SetTreeIndex(0);
         entry.fTreeIndex = index;
      }
      if (!index || index->IsZombie() || index->GetN() == 0) {
         DeleteIndices();
         MakeZombie();
         Error("TChainIndex", "Error creating a tree index on a tree in the chain");
         return;
      }

      TTreeIndex *ti_index = dynamic_cast<TTreeIndex*>(index);
      if (ti_index == 0) {
         Error("TChainIndex", "The underlying TTree must have a TTreeIndex but has a %s.",
               index->IsA()->GetName());
         return;
      }

      entry.fMinIndexValue = ti_index->GetIndexValues()[0];
      entry.fMaxIndexValue = ti_index->GetIndexValues()[index->GetN() - 1];
      fEntries.push_back(entry);
   }

   // Check if the indices of different trees are in order. If not then return an error.
   for (i = 0; i < Int_t(fEntries.size() - 1); i++) {
      if (fEntries[i].fMaxIndexValue > fEntries[i+1].fMinIndexValue) {
         DeleteIndices();
         MakeZombie();
         Error("TChainIndex", "The indices in files of this chain aren't sorted.");
      }
   }

}

//______________________________________________________________________________
void TChainIndex::Append(const TVirtualIndex *index, Bool_t delaySort )
{
   // add an index to this chain
   // if delaySort is kFALSE (default) check if the indices of different trees are in order.
   if (index) {
      const TTreeIndex *ti_index = dynamic_cast<const TTreeIndex*>(index);
      if (ti_index == 0) {
         Error("Append", "The given index is not a TTreeIndex but a %s",
               index->IsA()->GetName());
      }
      
      TChainIndexEntry entry;
      entry.fTreeIndex = 0;
      entry.fMinIndexValue = ti_index->GetIndexValues()[0];
      entry.fMaxIndexValue = ti_index->GetIndexValues()[index->GetN() - 1];
      fEntries.push_back(entry);
   }
   
   if (!delaySort) {
      // Check if the indices of different trees are in order. If not then return an error.
      for (Int_t i = 0; i < Int_t(fEntries.size() - 1); i++) {
         if (fEntries[i].fMaxIndexValue > fEntries[i+1].fMinIndexValue) {
            DeleteIndices();
            MakeZombie();
            Error("Append", "The indices in files of this chain aren't sorted.");
         }
      }
   }
}

//______________________________________________________________________________
void TChainIndex::DeleteIndices()
{
   // Delete all the indices which were built by this object
   for (unsigned int i = 0; i < fEntries.size(); i++) {
      if (fEntries[i].fTreeIndex) {
         if (fTree->GetTree() && fTree->GetTree()->GetTreeIndex() == fEntries[i].fTreeIndex) {
            fTree->GetTree()->SetTreeIndex(0);
            SafeDelete(fEntries[i].fTreeIndex);
         }
         SafeDelete(fEntries[i].fTreeIndex);
      }
   }
}

//______________________________________________________________________________
TChainIndex::~TChainIndex()
{
   // The destructor.
   DeleteIndices();
   if (fTree && fTree->GetTreeIndex() == this)
      fTree->SetTreeIndex(0);
}

//______________________________________________________________________________
std::pair<TVirtualIndex*, Int_t> TChainIndex::GetSubTreeIndex(Int_t major, Int_t minor) const
{
   // Returns a TVirtualIndex for a tree which holds the entry with the specified
   // major and minor values and the number of that tree.
   // If the index for that tree was created by this object it's set to the tree.
   // The tree index should be later released using ReleaseSubTreeIndex();

   using namespace std;
   if (fEntries.size() == 0) {
      Warning("GetSubTreeIndex", "No subindices in the chain. The chain is probably empty");
      return make_pair(static_cast<TVirtualIndex*>(0), 0);
   }

   Long64_t indexValue = (Long64_t(major) << 31) + minor;

   if (indexValue < fEntries[0].fMinIndexValue) {
      Warning("GetSubTreeIndex", "The index value is less than the smallest index values in subtrees");
      return make_pair(static_cast<TVirtualIndex*>(0), 0);
   }

   Int_t treeNo = fEntries.size() - 1;
   for (unsigned int i = 0; i < fEntries.size() - 1; i++) {
      if (indexValue < fEntries[i+1].fMinIndexValue) {
         treeNo = i;
         break;
      }
   }
   // Double check we found the right range.
   if (indexValue > fEntries[treeNo].fMaxIndexValue) {
      return make_pair(static_cast<TVirtualIndex*>(0), 0);
   }
   TChain* chain = dynamic_cast<TChain*> (fTree);
   R__ASSERT(chain);
   chain->LoadTree(chain->GetTreeOffset()[treeNo]);
   TVirtualIndex* index =  fTree->GetTree()->GetTreeIndex();
   if (index)
      return make_pair(static_cast<TVirtualIndex*>(index), treeNo);
   else {
      index = fEntries[treeNo].fTreeIndex;
      if (!index) {
         Warning("GetSubTreeIndex", "The tree has no index and the chain index"
                  " doesn't store an index for that tree");
         return make_pair(static_cast<TVirtualIndex*>(0), 0);
      }
      else {
         fTree->GetTree()->SetTreeIndex(index);
         return make_pair(static_cast<TVirtualIndex*>(index), treeNo);
      }
   }
}

//______________________________________________________________________________
void TChainIndex::ReleaseSubTreeIndex(TVirtualIndex* index, int treeNo) const
{
   // Releases the tree index got using GetSubTreeIndex. If the index was
   // created by this object it is removed from the current tree, so that it isn't
   // deleted in its destructor.

   if (fEntries[treeNo].fTreeIndex == index) {
      R__ASSERT(fTree->GetTree()->GetTreeIndex() == index);
      fTree->GetTree()->SetTreeIndex(0);
   }
}

//______________________________________________________________________________
Long64_t TChainIndex::GetEntryNumberFriend(const TTree *parent)
{
   // see TTreeIndex::GetEntryNumberFriend for description

   if (!parent) return -3;
   GetMajorFormulaParent(parent);
   GetMinorFormulaParent(parent);
   if (!fMajorFormulaParent || !fMinorFormulaParent) return -1;
   if (!fMajorFormulaParent->GetNdim() || !fMinorFormulaParent->GetNdim()) {
      // The Tree Index in the friend has a pair majorname,minorname
      // not available in the parent Tree T.
      // if the friend Tree has less entries than the parent, this is an error
      Long64_t pentry = parent->GetReadEntry();
      if (pentry >= fTree->GetEntries()) return -2;
      // otherwise we ignore the Tree Index and return the entry number
      // in the parent Tree.
      return pentry;
   }

   // majorname, minorname exist in the parent Tree
   // we find the current values pair majorv,minorv in the parent Tree
   Double_t majord = fMajorFormulaParent->EvalInstance();
   Double_t minord = fMinorFormulaParent->EvalInstance();
   Long64_t majorv = (Long64_t)majord;
   Long64_t minorv = (Long64_t)minord;
   // we check if this pair exist in the index.
   // if yes, we return the corresponding entry number
   // if not the function returns -1
   return fTree->GetEntryNumberWithIndex(majorv,minorv);
}

//______________________________________________________________________________
Long64_t TChainIndex::GetEntryNumberWithBestIndex(Int_t major, Int_t minor) const
{
   // See TTreeIndex::GetEntryNumberWithBestIndex for details.

   std::pair<TVirtualIndex*, Int_t> indexAndNumber = GetSubTreeIndex(major, minor);
   if (!indexAndNumber.first) {
      // Error("GetEntryNumberWithBestIndex","no index found");
      return -1;
   }
   else {
      Long64_t rv = indexAndNumber.first->GetEntryNumberWithBestIndex(major, minor);
      ReleaseSubTreeIndex(indexAndNumber.first, indexAndNumber.second);
      TChain* chain = dynamic_cast<TChain*> (fTree);
      R__ASSERT(chain);
      return rv + chain->GetTreeOffset()[indexAndNumber.second];
   }
}

//______________________________________________________________________________
Long64_t TChainIndex::GetEntryNumberWithIndex(Int_t major, Int_t minor) const
{
   // Returns the entry number with given index values.
   // See TTreeIndex::GetEntryNumberWithIndex for details.

   std::pair<TVirtualIndex*, Int_t> indexAndNumber = GetSubTreeIndex(major, minor);
   if (!indexAndNumber.first) {
      // Error("GetEntryNumberWithIndex","no index found");
      return -1;
   }
   else {
      Long64_t rv = indexAndNumber.first->GetEntryNumberWithIndex(major, minor);
      ReleaseSubTreeIndex(indexAndNumber.first, indexAndNumber.second);
      TChain* chain = dynamic_cast<TChain*> (fTree);
      R__ASSERT(chain);
      if (rv >= 0) {
         return rv + chain->GetTreeOffset()[indexAndNumber.second];
      } else {
         return rv;
      }
   }
}

//______________________________________________________________________________
TTreeFormula *TChainIndex::GetMajorFormulaParent(const TTree *parent)
{
   // return a pointer to the TreeFormula corresponding to the majorname in parent tree T

   if (!fMajorFormulaParent) {
      TTree::TFriendLock friendlock(fTree, TTree::kFindLeaf | TTree::kFindBranch | TTree::kGetBranch | TTree::kGetLeaf);
      fMajorFormulaParent = new TTreeFormula("MajorP",fMajorName.Data(),const_cast<TTree*>(parent));
      fMajorFormulaParent->SetQuickLoad(kTRUE);
   }
   if (fMajorFormulaParent->GetTree() != parent) {
      fMajorFormulaParent->SetTree(const_cast<TTree*>(parent));
      fMajorFormulaParent->UpdateFormulaLeaves();
   }
   return fMajorFormulaParent;
}

//______________________________________________________________________________
TTreeFormula *TChainIndex::GetMinorFormulaParent(const TTree *parent)
{
   // return a pointer to the TreeFormula corresponding to the minorname in parent tree T

   if (!fMinorFormulaParent) {
      // Prevent TTreeFormula from finding any of the branches in our TTree even if it
      // is a friend of the parent TTree.
      TTree::TFriendLock friendlock(fTree, TTree::kFindLeaf | TTree::kFindBranch | TTree::kGetBranch | TTree::kGetLeaf);
      fMinorFormulaParent = new TTreeFormula("MinorP",fMinorName.Data(),const_cast<TTree*>(parent));
      fMinorFormulaParent->SetQuickLoad(kTRUE);
   }
   if (fMinorFormulaParent->GetTree() != parent) {
      fMinorFormulaParent->SetTree(const_cast<TTree*>(parent));
      fMinorFormulaParent->UpdateFormulaLeaves();
   }

   return fMinorFormulaParent;
}

//______________________________________________________________________________
void TChainIndex::UpdateFormulaLeaves(const TTree *parent)
{
   // Updates the parent formulae.
   // Called by TChain::LoadTree when the parent chain changes it's tree.
   if (fMajorFormulaParent) { 
      // Prevent TTreeFormula from finding any of the branches in our TTree even if it
      // is a friend of the parent TTree.
      TTree::TFriendLock friendlock(fTree, TTree::kFindLeaf | TTree::kFindBranch | TTree::kGetBranch | TTree::kGetLeaf);
      if (parent) fMajorFormulaParent->SetTree((TTree*)parent);
      fMajorFormulaParent->UpdateFormulaLeaves();
   }
   if (fMinorFormulaParent) { 
      if (parent) fMinorFormulaParent->SetTree((TTree*)parent);
      fMinorFormulaParent->UpdateFormulaLeaves();
   }
}

//______________________________________________________________________________
void TChainIndex::SetTree(const TTree *T)
{
   // See TTreeIndex::SetTree.

   R__ASSERT(fTree == 0 || fTree == T || T==0);
}

