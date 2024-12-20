// @(#)root/base:$Id$
// Author: Philippe Canal  13/05/2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun, Fons Rademakers and al.           *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TFriendProxy
Concrete implementation of the proxy around a Friend Tree.
*/

#include "TChain.h"
#include "TFriendProxy.h"
#include "TTree.h"
#include "TList.h"
#include "TFriendElement.h"

ClassImp(ROOT::Internal::TFriendProxy);

namespace ROOT {
namespace Internal {

/////////////////////////////////////////////////////////////////////////////

TFriendProxy::TFriendProxy() : fDirector(nullptr,-1), fIndex(-1)
{
}

   /////////////////////////////////////////////////////////////////////////////
   /// Constructor.

   TFriendProxy::TFriendProxy(TBranchProxyDirector *director, TTree *main, Int_t index) :
      fDirector(nullptr,-1), fIndex(index)
   {

      auto getFriendAtIdx = [](TTree *dataset, Int_t frIdx) -> TTree * {
         if (!dataset)
            return nullptr;

         auto *friends = dataset->GetListOfFriends();
         if (!friends)
            return nullptr;

         auto *friendAtIdx = friends->At(frIdx);
         if (!friendAtIdx)
            return nullptr;

         auto *frEl = dynamic_cast<TFriendElement *>(friendAtIdx);
         if (!frEl)
            return nullptr;

         return frEl->GetTree();
      };

      // The list of friends needs to be accessed via GetTree()->GetListOfFriends()
      // (and not directly GetListOfFriends()), otherwise when `main` is a TChain we
      // might not recover the list correctly (see #6993 for the TTreeReader issue
      // and #6741 for a more complete discussion/explanation).
      if (main) {
         // But, we do need to get the top-level friend from the dataset (TTree or TChain)
         // to check if it has an index
         auto *topLevelFriendTree = getFriendAtIdx(main, fIndex);
         if (topLevelFriendTree) {
            fHasIndex = (topLevelFriendTree->GetTreeIndex() != nullptr);
         }

         // Then, point the director to the current tree from the friend that is being read
         auto *localFriendTree = getFriendAtIdx(main->GetTree(), fIndex);
         if (localFriendTree) {
            fDirector.SetTree(localFriendTree);
            // If we still do not think the friend has a TTreeIndex, retry for good measure
            if (!fHasIndex)
               fHasIndex = (localFriendTree->GetTreeIndex() != nullptr);
         }
      }

      director->Attach(this);
   }

   /////////////////////////////////////////////////////////////////////////////
   /// Return the entry number currently being looked at.

   Long64_t TFriendProxy::GetReadEntry() const
   {
      return fDirector.GetReadEntry();
   }

   /////////////////////////////////////////////////////////////////////////////
   /// Refresh the cached read entry number from the original tree.

   void TFriendProxy::ResetReadEntry()
   {
      // The 2nd call to GetTree is to insure we get the 'local' tree's entry in the case of a
      // chain.
      if (fDirector.GetTree()) fDirector.SetReadEntry(fDirector.GetTree()->GetTree()->GetReadEntry());
   }

  //////////////////////////////////////////////////////////////////////////////
  /// Update the address of the underlying tree.

   void TFriendProxy::Update(TTree *newmain)
   {
      if (!newmain)
         return;

      if (auto friends = newmain->GetTree()->GetListOfFriends()) {
         auto *element = dynamic_cast<TFriendElement *>(friends->At(fIndex));
         fDirector.SetTree(element ? element->GetTree() : nullptr);
      } else {
         fDirector.SetTree(nullptr);
      }
   }

} // namespace Internal
} // namespace ROOT
