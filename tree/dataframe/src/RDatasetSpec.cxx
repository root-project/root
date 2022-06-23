// Author: Vincenzo Eduardo Padulano CERN/UPV, Ivan Kabadzhov CERN  06/2022

/*************************************************************************
 * Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RDF/RDatasetSpec.hxx"

namespace ROOT {

namespace RDF {

RDatasetSpec::RDatasetSpec(const std::string &treeName, const std::string &fileNameGlob, const REntryRange &entryRange)
   : fTreeNames(std::vector<std::string>{std::move(treeName)}),
     fFileNameGlobs(std::vector<std::string>{std::move(fileNameGlob)}), fStartEntry(entryRange.fStartEntry),
     fEndEntry(entryRange.fEndEntry)
{
}

RDatasetSpec::RDatasetSpec(const std::string &treeName, const std::vector<std::string> &fileNameGlobs,
                           const REntryRange &entryRange)
   : fTreeNames(std::vector<std::string>{std::move(treeName)}), fFileNameGlobs(std::move(fileNameGlobs)),
     fStartEntry(entryRange.fStartEntry), fEndEntry(entryRange.fEndEntry)
{
}

RDatasetSpec::RDatasetSpec(const std::vector<std::pair<std::string, std::string>> &treeAndFileNameGlobs,
                           const REntryRange &entryRange)
   : fStartEntry(entryRange.fStartEntry), fEndEntry(entryRange.fEndEntry)
{
   fTreeNames.reserve(treeAndFileNameGlobs.size());
   fFileNameGlobs.reserve(treeAndFileNameGlobs.size());
   for (auto &p : treeAndFileNameGlobs) {
      fTreeNames.emplace_back(std::move(p.first));
      fFileNameGlobs.emplace_back(std::move(p.second));
   }
}

void RDatasetSpec::AddFriend(const std::string &treeName, const std::string &fileNameGlob, const std::string &alias)
{
   fFriendInfo.fFriendNames.emplace_back(std::make_pair(std::move(treeName), std::move(alias)));
   fFriendInfo.fFriendFileNames.emplace_back(std::vector<std::string>{std::move(fileNameGlob)});
   fFriendInfo.fFriendChainSubNames.emplace_back(); // this is a tree
}

void RDatasetSpec::AddFriend(const std::string &treeName, const std::vector<std::string> &fileNameGlobs,
                             const std::string &alias)
{
   fFriendInfo.fFriendNames.emplace_back(std::make_pair("", std::move(alias)));
   fFriendInfo.fFriendFileNames.emplace_back(std::move(fileNameGlobs));
   fFriendInfo.fFriendChainSubNames.emplace_back(std::vector<std::string>(fileNameGlobs.size(), std::move(treeName)));
}

void RDatasetSpec::AddFriend(const std::vector<std::pair<std::string, std::string>> &treeAndFileNameGlobs,
                             const std::string &alias)
{
   fFriendInfo.fFriendNames.emplace_back(std::make_pair("", std::move(alias)));
   fFriendInfo.fFriendFileNames.emplace_back();
   fFriendInfo.fFriendChainSubNames.emplace_back();
   auto &fileNames = fFriendInfo.fFriendFileNames.back();
   auto &chainSubNames = fFriendInfo.fFriendChainSubNames.back();
   fileNames.reserve(treeAndFileNameGlobs.size());
   chainSubNames.reserve(treeAndFileNameGlobs.size());
   for (auto &p : treeAndFileNameGlobs) {
      chainSubNames.emplace_back(std::move(p.first));
      fileNames.emplace_back(std::move(p.second));
   }
}

} // namespace RDF
} // namespace ROOT