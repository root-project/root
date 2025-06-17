// @(#)root/tree:$Id$
// Author: Marek Biskup   07/06/2005

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TChainIndex
A Chain Index.
A Chain Index with majorname and minorname.
It uses tree indices of all the trees in the chain instead of building
a new index.
The index values from the first tree should be less then
all the index values from the second tree, and so on.
If a tree in the chain doesn't have an index the index will be created
and kept inside this chain index.
*/

#include "TChainIndex.h"
#include "TChain.h"
#include "TTreeFormula.h"
#include "TTreeIndex.h"
#include "TFile.h"
#include "TError.h"

#include <cstring> // std::strlen

////////////////////////////////////////////////////////////////////////////////
/// \class TChainIndex::TChainIndexEntry
/// Holds a description of indices of trees in the chain.

void TChainIndex::TChainIndexEntry::SetMinMaxFrom(const TTreeIndex *index )
{
   fMinIndexValue    = index->GetIndexValues()[0];
   fMinIndexValMinor = index->GetIndexValuesMinor()[0];
   fMaxIndexValue    = index->GetIndexValues()[index->GetN() - 1];
   fMaxIndexValMinor = index->GetIndexValuesMinor()[index->GetN() - 1];
}

void TChainIndex::TChainIndexEntry::Swap(TChainIndex::TChainIndexEntry &other)
{
   std::swap(fMinIndexValue, other.fMinIndexValue);
   std::swap(fMinIndexValMinor, other.fMinIndexValMinor);
   std::swap(fMaxIndexValue, other.fMaxIndexValue);
   std::swap(fMaxIndexValMinor, other.fMaxIndexValMinor);
   std::swap(fTreeIndex, other.fTreeIndex);
}

TChainIndex::TChainIndexEntry::TChainIndexEntry(const TChainIndex::TChainIndexEntry &other)
{
   fMinIndexValue = other.fMinIndexValue;
   fMinIndexValMinor = other.fMinIndexValMinor;
   fMaxIndexValue = other.fMaxIndexValue;
   fMaxIndexValMinor = other.fMaxIndexValMinor;
   fTreeIndex = (other.fTreeIndex ? static_cast<TVirtualIndex *>(other.fTreeIndex->Clone()) : nullptr);
}

ClassImp(TChainIndex);

////////////////////////////////////////////////////////////////////////////////
/// Default constructor for TChainIndex

TChainIndex::TChainIndex(): TVirtualIndex()
{
   fTree = nullptr;
   fMajorFormulaParent = fMinorFormulaParent = nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Normal constructor for TChainIndex. See TTreeIndex::TTreeIndex for the description of the
/// parameters.
/// The tree must be a TChain.
/// All the index values in the first tree of the chain must be
/// less then any index value in the second one, and so on.
/// If any of those requirements isn't met the object becomes a zombie.
/// If some subtrees don't have indices the indices are created and stored inside this
/// TChainIndex.
/// In some cases, a warning is printed about unsorted indices in the TChain files. Note
/// that unsorted indices lead to a significant performance degradation, not only when building the index itself,
/// but also later on when performing the joining with other datasets. Thus, in general, it is not recommended to
/// ignore this warning except for special cases with prior knowledge that sorting the files and/or entries is actually
/// more expensive, or just not possible.

TChainIndex::TChainIndex(const TTree *T, const char *majorname, const char *minorname)
           : TVirtualIndex()
{
   fTree = nullptr;
   fMajorFormulaParent = fMinorFormulaParent = nullptr;

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

   // Go through all the trees and check if they have indices. If not then build them.
   for (i = 0; i < chain->GetNtrees(); i++) {
      chain->LoadTree((chain->GetTreeOffset())[i]);
      TVirtualIndex *index = chain->GetTree()->GetTreeIndex();

      TChainIndexEntry entry;
      entry.fTreeIndex = nullptr;

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
         chain->GetTree()->SetTreeIndex(nullptr);
         entry.fTreeIndex = index;
      }
      if (!index || index->IsZombie() || index->GetN() == 0) {
         DeleteIndices();
         MakeZombie();
         Error("TChainIndex", "Error creating a tree index on a tree in the chain");
         return;
      }

      TTreeIndex *ti_index = dynamic_cast<TTreeIndex*>(index);
      if (ti_index == nullptr) {
         Error("TChainIndex", "The underlying TTree must have a TTreeIndex but has a %s.",
               index->IsA()->GetName());
         return;
      }

      entry.SetMinMaxFrom(ti_index);
      fEntries.push_back(entry);
   }

   // Check if the indices of different trees are in order. If not then return an error.
   for (i = 0; i < Int_t(fEntries.size() - 1); i++) {
      if( fEntries[i].GetMaxIndexValPair() > fEntries[i+1].GetMinIndexValPair() ) {
         DeleteIndices();
         MakeZombie();
         Warning("TChainIndex", "The indices in files of this chain aren't sorted.");
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Add an index to this chain.
/// if delaySort is false (default) check if the indices of different trees are in order.

void TChainIndex::Append(const TVirtualIndex *index, bool delaySort )
{
   if (index) {
      const TTreeIndex *ti_index = dynamic_cast<const TTreeIndex*>(index);
      if (ti_index == nullptr) {
         Error("Append", "The given index is not a TTreeIndex but a %s",
               index->IsA()->GetName());
      }

      TChainIndexEntry entry;
      entry.fTreeIndex = nullptr;
      entry.SetMinMaxFrom(ti_index);
      fEntries.push_back(entry);
   }

   if (!delaySort) {
      // Check if the indices of different trees are in order. If not then return an error.
      for (Int_t i = 0; i < Int_t(fEntries.size() - 1); i++) {
         if( fEntries[i].GetMaxIndexValPair() > fEntries[i+1].GetMinIndexValPair() ) {
            DeleteIndices();
            MakeZombie();
            Error("Append", "The indices in files of this chain aren't sorted.");
         }
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Delete all the indices which were built by this object.

void TChainIndex::DeleteIndices()
{
   for (unsigned int i = 0; i < fEntries.size(); i++) {
      if (fEntries[i].fTreeIndex) {
         if (fTree->GetTree() && fTree->GetTree()->GetTreeIndex() == fEntries[i].fTreeIndex) {
            fTree->GetTree()->SetTreeIndex(nullptr);
            SafeDelete(fEntries[i].fTreeIndex);
         }
         SafeDelete(fEntries[i].fTreeIndex);
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// The destructor.

TChainIndex::~TChainIndex()
{
   if (fTree) {
      DeleteIndices();
      if (fTree->GetTreeIndex() == this)
         fTree->SetTreeIndex(nullptr);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Returns a TVirtualIndex for a tree which holds the entry with the specified
/// major and minor values and the number of that tree.
/// If the index for that tree was created by this object it's set to the tree.
/// The tree index should be later released using ReleaseSubTreeIndex();

std::pair<TVirtualIndex*, Int_t> TChainIndex::GetSubTreeIndex(Long64_t major, Long64_t minor) const
{
   using std::make_pair;
   if (fEntries.empty()) {
      Warning("GetSubTreeIndex", "No subindices in the chain. The chain is probably empty");
      return make_pair(static_cast<TVirtualIndex*>(nullptr), 0);
   }

   const TChainIndexEntry::IndexValPair_t     indexValue(major, minor);

   if( indexValue < fEntries[0].GetMinIndexValPair() ) {
      Warning("GetSubTreeIndex", "The index value is less than the smallest index values in subtrees");
      return make_pair(static_cast<TVirtualIndex*>(nullptr), 0);
   }

   Int_t treeNo = fEntries.size() - 1;
   for (unsigned int i = 0; i < fEntries.size() - 1; i++) {
      if( indexValue < fEntries[i+1].GetMinIndexValPair() ) {
         treeNo = i;
         break;
      }
   }
   // Double check we found the right range.
   if( indexValue > fEntries[treeNo].GetMaxIndexValPair() ) {
      return make_pair(static_cast<TVirtualIndex*>(nullptr), 0);
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
         return make_pair(static_cast<TVirtualIndex*>(nullptr), 0);
      }
      else {
         fTree->GetTree()->SetTreeIndex(index);
         return make_pair(static_cast<TVirtualIndex*>(index), treeNo);
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Releases the tree index got using GetSubTreeIndex. If the index was
/// created by this object it is removed from the current tree, so that it isn't
/// deleted in its destructor.

void TChainIndex::ReleaseSubTreeIndex(TVirtualIndex* index, int treeNo) const
{
   if (fEntries[treeNo].fTreeIndex == index) {
      R__ASSERT(fTree->GetTree()->GetTreeIndex() == index);
      fTree->GetTree()->SetTreeIndex(nullptr);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// See TTreeIndex::GetEntryNumberFriend for description

Long64_t TChainIndex::GetEntryNumberFriend(const TTree *parent)
{
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

////////////////////////////////////////////////////////////////////////////////
/// See TTreeIndex::GetEntryNumberWithBestIndex for details.

Long64_t TChainIndex::GetEntryNumberWithBestIndex(Long64_t major, Long64_t minor) const
{
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

////////////////////////////////////////////////////////////////////////////////
/// Returns the entry number with given index values.
/// See TTreeIndex::GetEntryNumberWithIndex for details.

Long64_t TChainIndex::GetEntryNumberWithIndex(Long64_t major, Long64_t minor) const
{
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

////////////////////////////////////////////////////////////////////////////////
/// Return a pointer to the TreeFormula corresponding to the majorname in parent tree T.

TTreeFormula *TChainIndex::GetMajorFormulaParent(const TTree *parent)
{
   if (!fMajorFormulaParent) {
      TTree::TFriendLock friendlock(fTree, TTree::kFindLeaf | TTree::kFindBranch | TTree::kGetBranch | TTree::kGetLeaf);
      fMajorFormulaParent = new TTreeFormula("MajorP",fMajorName.Data(),const_cast<TTree*>(parent));
      fMajorFormulaParent->SetQuickLoad(true);
   }
   if (fMajorFormulaParent->GetTree() != parent) {
      fMajorFormulaParent->SetTree(const_cast<TTree*>(parent));
      fMajorFormulaParent->UpdateFormulaLeaves();
   }
   return fMajorFormulaParent;
}

////////////////////////////////////////////////////////////////////////////////
/// Return a pointer to the TreeFormula corresponding to the minorname in parent tree T.

TTreeFormula *TChainIndex::GetMinorFormulaParent(const TTree *parent)
{
   if (!fMinorFormulaParent) {
      // Prevent TTreeFormula from finding any of the branches in our TTree even if it
      // is a friend of the parent TTree.
      TTree::TFriendLock friendlock(fTree, TTree::kFindLeaf | TTree::kFindBranch | TTree::kGetBranch | TTree::kGetLeaf);
      fMinorFormulaParent = new TTreeFormula("MinorP",fMinorName.Data(),const_cast<TTree*>(parent));
      fMinorFormulaParent->SetQuickLoad(true);
   }
   if (fMinorFormulaParent->GetTree() != parent) {
      fMinorFormulaParent->SetTree(const_cast<TTree*>(parent));
      fMinorFormulaParent->UpdateFormulaLeaves();
   }

   return fMinorFormulaParent;
}

////////////////////////////////////////////////////////////////////////////////
/// Return true if index can be applied to the TTree

bool TChainIndex::IsValidFor(const TTree *parent)
{
   auto *majorFormula = GetMajorFormulaParent(parent);
   auto *minorFormula = GetMinorFormulaParent(parent);
   if ((majorFormula == nullptr || majorFormula->GetNdim() == 0) ||
       (minorFormula == nullptr || minorFormula->GetNdim() == 0))
         return false;
   return true;
}

////////////////////////////////////////////////////////////////////////////////
/// Updates the parent formulae.
/// Called by TChain::LoadTree when the parent chain changes it's tree.

void TChainIndex::UpdateFormulaLeaves(const TTree *parent)
{
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

////////////////////////////////////////////////////////////////////////////////
/// See TTreeIndex::SetTree.

void TChainIndex::SetTree(TTree *T)
{
   R__ASSERT(fTree == nullptr || fTree == T || T==nullptr);
   fTree = T;
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Create a deep copy of the TChainIndex
/// \param[in] newname A new name for the index
///
/// The new index is allocated on the heap without being managed. Also, it is
/// not attached to any tree. It is the responsibility of the caller to manage
/// its lifetime and attach it to a tree if necessary.
TObject *TChainIndex::Clone(const char *newname) const
{
   auto index = new TChainIndex();
   index->SetName(newname && std::strlen(newname) ? newname : GetName());
   index->SetTitle(GetTitle());

   // Note that the TTreeFormula * data members are not cloned since they would
   // need the attached tree data member to function properly.
   index->fMajorName = fMajorName;
   index->fMinorName = fMinorName;

   index->fEntries = fEntries;

   return index;
}
