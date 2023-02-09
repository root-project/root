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

RFriendInfo::RFriendInfo(const RFriendInfo &other)
{
   *this = other;
}

RFriendInfo &RFriendInfo::operator=(const RFriendInfo &other)
{
   fFriendNames = other.fFriendNames;
   fFriendFileNames = other.fFriendFileNames;
   fFriendChainSubNames = other.fFriendChainSubNames;
   fNEntriesPerTreePerFriend = other.fNEntriesPerTreePerFriend;

   for (const auto &idxInfo : other.fTreeIndexInfos)
      fTreeIndexInfos.emplace_back(static_cast<TVirtualIndex *>(idxInfo ? idxInfo->Clone() : nullptr));

   return *this;
}

/// Construct a RFriendInfo object from its components.
RFriendInfo::RFriendInfo(std::vector<std::pair<std::string, std::string>> friendNames,
                         std::vector<std::vector<std::string>> friendFileNames,
                         std::vector<std::vector<std::string>> friendChainSubNames,
                         std::vector<std::vector<std::int64_t>> nEntriesPerTreePerFriend,
                         std::vector<std::unique_ptr<TVirtualIndex>> treeIndexInfos)
   : fFriendNames(std::move(friendNames)),
     fFriendFileNames(std::move(friendFileNames)),
     fFriendChainSubNames(std::move(friendChainSubNames)),
     fNEntriesPerTreePerFriend(std::move(nEntriesPerTreePerFriend)),
     fTreeIndexInfos(std::move(treeIndexInfos))
{
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Add information of a single friend.
///
/// \param[in] treeName Name of the tree.
/// \param[in] fileNameGlob Path to the file. Refer to TChain::Add for globbing rules.
/// \param[in] alias Alias for this friend.
/// \param[in] nEntries Number of entries for this friend.
/// \param[in] indexInfo Tree index info for this friend.
void RFriendInfo::AddFriend(const std::string &treeName, const std::string &fileNameGlob, const std::string &alias,
                            std::int64_t nEntries, TVirtualIndex *indexInfo)
{
   fFriendNames.emplace_back(std::make_pair(treeName, alias));
   fFriendFileNames.emplace_back(std::vector<std::string>{fileNameGlob});
   fFriendChainSubNames.emplace_back();
   fNEntriesPerTreePerFriend.push_back(std::vector<std::int64_t>({nEntries}));
   fTreeIndexInfos.emplace_back(static_cast<TVirtualIndex *>(indexInfo ? indexInfo->Clone() : nullptr));
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Add information of a single friend.
///
/// \param[in] treeName Name of the tree.
/// \param[in] fileNameGlobs Paths to the files. Refer to TChain::Add for globbing rules.
/// \param[in] alias Alias for this friend.
/// \param[in] nEntriesVec Number of entries for each file of this friend.
/// \param[in] indexInfo Tree index info for this friend.
void RFriendInfo::AddFriend(const std::string &treeName, const std::vector<std::string> &fileNameGlobs,
                            const std::string &alias, const std::vector<std::int64_t> &nEntriesVec,
                            TVirtualIndex *indexInfo)
{
   fFriendNames.emplace_back(std::make_pair(treeName, alias));
   fFriendFileNames.emplace_back(fileNameGlobs);
   fFriendChainSubNames.emplace_back(std::vector<std::string>(fileNameGlobs.size(), treeName));
   fNEntriesPerTreePerFriend.push_back(
      nEntriesVec.empty() ? std::vector<int64_t>(fileNameGlobs.size(), std::numeric_limits<std::int64_t>::max())
                          : nEntriesVec);
   fTreeIndexInfos.emplace_back(static_cast<TVirtualIndex *>(indexInfo ? indexInfo->Clone() : nullptr));
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Add information of a single friend.
///
/// \param[in] treeAndFileNameGlobs Pairs of (treename, filename). Refer to TChain::Add for globbing rules.
/// \param[in] alias Alias for this friend.
/// \param[in] nEntriesVec Number of entries for each file of this friend.
/// \param[in] indexInfo Tree index info for this friend.
void RFriendInfo::AddFriend(const std::vector<std::pair<std::string, std::string>> &treeAndFileNameGlobs,
                            const std::string &alias, const std::vector<std::int64_t> &nEntriesVec,
                            TVirtualIndex *indexInfo)
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
   fTreeIndexInfos.emplace_back(static_cast<TVirtualIndex *>(indexInfo ? indexInfo->Clone() : nullptr));
}

} // namespace TreeUtils
} // namespace ROOT
