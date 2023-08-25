/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/**
 \file ROOT/InternalTreeUtils.hxx
 \ingroup tree
 \author Enric Tejedor Saavedra
 \author Enrico Guiraud
 \author Vincenzo Eduardo Padulano
 \date 2021-03
*/

#ifndef ROOT_INTERNAL_TREEUTILS_H
#define ROOT_INTERNAL_TREEUTILS_H

#include "TTree.h"
#include "TChain.h"
#include "TNotifyLink.h"
#include "TObjArray.h"
#include "ROOT/RFriendInfo.hxx"

#include <memory>
#include <string>
#include <utility> // std::pair
#include <vector>

namespace ROOT {
namespace Internal {
/**
\namespace ROOT::Internal::TreeUtils
\ingroup tree
\brief Namespace hosting functions and classes to retrieve tree information for internal use.
*/
namespace TreeUtils {

std::vector<std::string> GetTopLevelBranchNames(TTree &t);
std::vector<std::string> GetFileNamesFromTree(const TTree &tree);
ROOT::TreeUtils::RFriendInfo GetFriendInfo(const TTree &tree, bool retrieveEntries = false);
std::vector<std::string> GetTreeFullPaths(const TTree &tree);

void ClearMustCleanupBits(TObjArray &arr);

class RNoCleanupNotifierHelper {
   TChain *fChain = nullptr;

public:
   bool Notify()
   {
      TTree *t = fChain->GetTree();
      TObjArray *branches = t->GetListOfBranches();
      ClearMustCleanupBits(*branches);
      return true;
   }

   void RegisterChain(TChain *c) { fChain = c; }
};

class RNoCleanupNotifier : public TNotifyLink<RNoCleanupNotifierHelper> {
   RNoCleanupNotifierHelper fNoCleanupNotifierHelper;

public:
   RNoCleanupNotifier() : TNotifyLink<RNoCleanupNotifierHelper>(&fNoCleanupNotifierHelper) {}

   void RegisterChain(TChain &c)
   {
      fNoCleanupNotifierHelper.RegisterChain(&c);
      this->PrependLink(c);
   }

   ClassDefOverride(RNoCleanupNotifier, 0);
};

std::unique_ptr<TChain> MakeChainForMT(const std::string &name = "", const std::string &title = "");
std::vector<std::unique_ptr<TChain>> MakeFriends(const ROOT::TreeUtils::RFriendInfo &finfo);

std::vector<std::string> ExpandGlob(const std::string &glob);

} // namespace TreeUtils
} // namespace Internal
} // namespace ROOT

#endif // ROOT_INTERNAL_TREEUTILS_H
