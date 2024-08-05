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

      auto getNTrees = [](const TTree &dataset) {
         try {
            const auto &chain = dynamic_cast<const TChain &>(dataset);
            return chain.GetNtrees();
         } catch (const std::bad_cast &) {
            return 1;
         }
      };

      // The list of friends needs to be accessed via GetTree()->GetListOfFriends()
      // (and not directly GetListOfFriends()), otherwise when `main` is a TChain we
      // might not recover the list correctly (see #6993 for the TTreeReader issue
      // and #6741 for a more complete discussion/explanation).
      if (main) {
         // But, we do need to get the top-level friend from the dataset (TTree or TChain)
         // to retrieve the correct number of trees in this friend.
         auto *topLevelFriendTree = getFriendAtIdx(main, fIndex);
         if (topLevelFriendTree) {
            fNTrees = getNTrees(*topLevelFriendTree);
            fIsIndexed = topLevelFriendTree->GetTreeIndex() ? true : false;
         }

         // Then, point the director to the current tree from the friend that is being read
         auto *localFriendTree = getFriendAtIdx(main->GetTree(), fIndex);
         if (localFriendTree) {
            fDirector.SetTree(localFriendTree);
            // if fNTrees is still equal to zero it means we could not find
            // a top-level friend for the input tree. One possible situation
            // where this may occur is when a friend tree is autoloaded from
            // the same TFile where the main tree is stored
            if (fNTrees == 0)
               fNTrees = getNTrees(*localFriendTree);
            // If we still do not think the friend has a TTreeIndex, retry for good measure
            if (!fIsIndexed)
               fIsIndexed = localFriendTree->GetTreeIndex() ? true : false;
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
      if (newmain && newmain->GetTree()->GetListOfFriends()) {
         TObject *obj = newmain->GetTree()->GetListOfFriends()->At(fIndex);
         TFriendElement *element = dynamic_cast<TFriendElement *>(obj);
         if (element) {
            // Since we know via TFriendProxy's constructor that the
            // TTree * being passed here corresponds to a local friend and
            // not to the "global" one (i.e. the top-level tree which may
            // be a TTree or a TChain), we have to use the information
            // coming from the chain offset (which is available both from
            // a TTree or a TChain) to keep track of how many trees we
            // have seen so far in this friend.
            if (fLastChainOffset == -1) {
               // First tree we see ever, reset the chain offset and update
               // the number of trees.
               fLastChainOffset++;
               fNTreesSoFar++;
            }
            // We have to make sure that the chain offset is actually greater than
            // the number we have stored previously. It may in fact happen that
            // the same friend tree appears more than once in an Update call.
            // For example, if the main chain is misaligned w.r.t. the friend
            // and we are changing a file of the main chain.
            if (auto chOffset = element->GetTree()->GetChainOffset(); chOffset > fLastChainOffset) {
               fLastChainOffset += chOffset;
               fNTreesSoFar++;
            }
            fDirector.SetTree(element->GetTree());
         } else {
            fDirector.SetTree(nullptr);
         }
      } else {
         fDirector.SetTree(nullptr);
      }
   }

} // namespace Internal
} // namespace ROOT
