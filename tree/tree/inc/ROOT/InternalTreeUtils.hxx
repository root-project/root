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

#include <utility> // std::pair
#include <vector>
#include <string>

class TTree;

namespace ROOT {
namespace Internal {
/**
\namespace ROOT::Internal::TreeUtils
\ingroup tree
\brief Namespace hosting functions and classes to retrieve tree information for internal use.
*/
namespace TreeUtils {

using NameAlias = std::pair<std::string, std::string>; ///< A pair of name and alias of a TTree's friend tree.
/**
\struct ROOT::Internal::TreeUtils::RFriendInfo
\brief Information about friend trees of a certain TTree or TChain object.
\ingroup tree
*/
struct RFriendInfo {

   std::vector<NameAlias> fFriendNames; ///< Pairs of names and aliases of friend trees/chains.
   /**
   Names of the files where each friend is stored. fFriendFileNames[i] is the
   list of files for friend with name fFriendNames[i].
   */
   std::vector<std::vector<std::string>> fFriendFileNames;
   /**
      Names of the subtrees of a friend TChain. fFriendChainSubNames[i] is the
      list of names of the trees that make a friend TChain whose information is
      stored at fFriendNames[i] and fFriendFileNames[i]. If instead the friend
      tree at position `i` is a TTree, fFriendChainSubNames[i] will be just a
      vector with a single empty string.
   */
   std::vector<std::vector<std::string>> fFriendChainSubNames;
};

std::vector<std::string> GetFileNamesFromTree(const TTree &tree);
RFriendInfo GetFriendInfo(const TTree &tree);
std::vector<std::string> GetTreeFullPaths(const TTree &tree);

} // namespace TreeUtils
} // namespace Internal
} // namespace ROOT

#endif // ROOT_INTERNAL_TREEUTILS_H
