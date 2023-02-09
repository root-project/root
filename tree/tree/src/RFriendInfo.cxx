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
/// \param[in] nEntries Number of entries for this friend.
void RFriendInfo::AddFriend(const std::string &treeName, const std::string &fileNameGlob, const std::string &alias,
                            std::int64_t nEntries)
{
   fFriendNames.emplace_back(std::make_pair(treeName, alias));
   fFriendFileNames.emplace_back(std::vector<std::string>{fileNameGlob});
   fFriendChainSubNames.emplace_back();
   fNEntriesPerTreePerFriend.push_back(std::vector<std::int64_t>({nEntries}));
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Add information of a single friend.
///
/// \param[in] treeName Name of the tree.
/// \param[in] fileNameGlobs Paths to the files. Refer to TChain::Add for globbing rules.
/// \param[in] alias Alias for this friend.
/// \param[in] nEntriesVec Number of entries for each file of this friend.
void RFriendInfo::AddFriend(const std::string &treeName, const std::vector<std::string> &fileNameGlobs,
                            const std::string &alias, const std::vector<std::int64_t> &nEntriesVec)
{
   fFriendNames.emplace_back(std::make_pair(treeName, alias));
   fFriendFileNames.emplace_back(fileNameGlobs);
   fFriendChainSubNames.emplace_back(std::vector<std::string>(fileNameGlobs.size(), treeName));
   fNEntriesPerTreePerFriend.push_back(
      nEntriesVec.empty() ? std::vector<int64_t>(fileNameGlobs.size(), std::numeric_limits<std::int64_t>::max())
                          : nEntriesVec);
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Add information of a single friend.
///
/// \param[in] treeAndFileNameGlobs Pairs of (treename, filename). Refer to TChain::Add for globbing rules.
/// \param[in] alias Alias for this friend.
/// \param[in] nEntriesVec Number of entries for each file of this friend.
void RFriendInfo::AddFriend(const std::vector<std::pair<std::string, std::string>> &treeAndFileNameGlobs,
                            const std::string &alias, const std::vector<std::int64_t> &nEntriesVec)
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
   fNEntriesPerTreePerFriend.push_back(
      nEntriesVec.empty() ? std::vector<int64_t>(treeAndFileNameGlobs.size(), std::numeric_limits<std::int64_t>::max())
                          : nEntriesVec);
}

} // namespace TreeUtils
} // namespace ROOT
