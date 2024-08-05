// @(#)root/treeplayer:$Id$
// Author: Philippe Canal 01/06/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers and al.        *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TFriendProxy
#define ROOT_TFriendProxy

#include "TBranchProxyDirector.h"

class TTree;

namespace ROOT {
namespace Internal {

   class TFriendProxy {
   protected:
      TBranchProxyDirector fDirector; ///< Contain pointer to TTree and entry to be read
      Int_t  fIndex;                  ///< Index of this tree in the list of friends
      bool fIsIndexed{false};         ///< Whether this friend has a TTreeIndex attached
      Int_t fNTrees{0};               ///< How many trees are in this friend dataset
      Int_t fNTreesSoFar{0};          ///< How many trees have been processed so far
      Long64_t fLastChainOffset{-1};  ///< Offset of the current tree in the chain

   public:
      TFriendProxy();
      TFriendProxy(TBranchProxyDirector *director, TTree *main, Int_t index);

      TBranchProxyDirector *GetDirector() { return &fDirector; }

      Long64_t GetReadEntry() const;
      void     ResetReadEntry();
      void     Update(TTree *newmain);
      bool IsIndexed() const { return fIsIndexed; }
      Int_t GetNTrees() const { return fNTrees; }
      Int_t GetNTreesSoFar() const { return fNTreesSoFar; }
   };

} // namespace Internal
} // namespace ROOT

#endif
