/*************************************************************************
 * Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RFriendInfo.hxx"
#include <iterator> // std::back_inserter

namespace ROOT {
namespace TreeUtils {

////////////////////////////////////////////////////////////////////////////////
/// \brief Add information of a single friend.
///
/// \param[in] treeName Name of the tree.
/// \param[in] fileNameGlob Path to the file. Refer to TChain::Add for globbing rules.
/// \param[in] alias Alias for this friend.
void RFriendInfo::AddFriend(const std::string &treeName, const std::string &fileNameGlob, const std::string &alias)
{
   fFriendNames.emplace_back(std::make_pair(treeName, alias));
   fFriendFileNames.emplace_back(std::vector<std::string>{fileNameGlob});
   fFriendChainSubNames.emplace_back();
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Add information of a single friend.
///
/// \param[in] treeName Name of the tree.
/// \param[in] fileNameGlobs Paths to the files. Refer to TChain::Add for globbing rules.
/// \param[in] alias Alias for this friend.
void RFriendInfo::AddFriend(const std::string &treeName, const std::vector<std::string> &fileNameGlobs,
                            const std::string &alias)
{
   fFriendNames.emplace_back(std::make_pair(treeName, alias));
   fFriendFileNames.emplace_back(fileNameGlobs);
   fFriendChainSubNames.emplace_back(std::vector<std::string>(fileNameGlobs.size(), treeName));
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Add information of a single friend.
///
/// \param[in] treeAndFileNameGlobs Pairs of (treename, filename). Refer to TChain::Add for globbing rules.
/// \param[in] alias Alias for this friend.
void RFriendInfo::AddFriend(const std::vector<std::pair<std::string, std::string>> &treeAndFileNameGlobs,
                            const std::string &alias)
{
   fFriendNames.emplace_back(std::make_pair("", alias));

   fFriendFileNames.emplace_back();
   fFriendChainSubNames.emplace_back();

   auto &theseFileNames = fFriendFileNames.back();
   auto &theseChainSubNames = fFriendChainSubNames.back();
   auto nPairs = treeAndFileNameGlobs.size();
   theseFileNames.reserve(nPairs);
   theseChainSubNames.reserve(nPairs);

   auto fSubNamesIt = std::back_inserter(theseChainSubNames);
   auto fNamesIt = std::back_inserter(theseFileNames);

   for (const auto &names : treeAndFileNameGlobs) {
      *fSubNamesIt = names.first;
      *fNamesIt = names.second;
   }
}

} // namespace TreeUtils
} // namespace ROOT
