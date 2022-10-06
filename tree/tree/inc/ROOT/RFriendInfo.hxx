/*************************************************************************
 * Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/**
 \file ROOT/RFriendInfo.hxx
 \ingroup tree
 \author Ivan Kabadzhov
 \author Enrico Guiraud
 \author Vincenzo Eduardo Padulano
 \date 2022-10
*/

#ifndef ROOT_RFRIENDINFO_H
#define ROOT_RFRIENDINFO_H

#include <string>
#include <utility> // std::pair
#include <vector>

class TTree;

namespace ROOT {
namespace TreeUtils {
/**
\struct ROOT::TreeUtils::RFriendInfo
\brief Information about friend trees of a certain TTree or TChain object.
\ingroup tree
*/
struct RFriendInfo {

   std::vector<std::pair<std::string, std::string>>
      fFriendNames; ///< Pairs of names and aliases of friend trees/chains.
   /**
   Names of the files where each friend is stored. fFriendFileNames[i] is the
   list of files for friend with name fFriendNames[i].
   */
   std::vector<std::vector<std::string>> fFriendFileNames;
   /**
      Names of the subtrees of a friend TChain. fFriendChainSubNames[i] is the
      list of names of the trees that make a friend TChain whose information is
      stored at fFriendNames[i] and fFriendFileNames[i]. If instead the friend
      tree at position `i` is a TTree, fFriendChainSubNames[i] will be an empty
      vector.
   */
   std::vector<std::vector<std::string>> fFriendChainSubNames;

   void AddFriend(const std::string &treeName, const std::string &fileNameGlob, const std::string &alias = "");

   void
   AddFriend(const std::string &treeName, const std::vector<std::string> &fileNameGlobs, const std::string &alias = "");

   void AddFriend(const std::vector<std::pair<std::string, std::string>> &treeAndFileNameGlobs,
                  const std::string &alias = "");
};

} // namespace TreeUtils
} // namespace ROOT

#endif // ROOT_RFRIENDINFO_H