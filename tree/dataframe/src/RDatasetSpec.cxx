// Author: Vincenzo Eduardo Padulano CERN/UPV, Ivan Kabadzhov CERN  06/2022

/*************************************************************************
 * Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RDF/RDatasetSpec.hxx"
#include <stdexcept> // std::logic_error

namespace ROOT {

namespace RDF {

namespace Experimental {

RDatasetSpec::REntryRange::REntryRange() {}

RDatasetSpec::REntryRange::REntryRange(Long64_t end) : fEnd(end) {}

RDatasetSpec::REntryRange::REntryRange(Long64_t begin, Long64_t end) : fBegin(begin), fEnd(end)
{
   if (fBegin > fEnd)
      throw std::logic_error("The starting entry cannot be larger than the ending entry in the "
                             "creation of a dataset specification.");
}

const std::vector<std::string> &RDatasetSpec::GetTreeNames() const
{
   return fTreeNames;
}

const std::vector<std::string> &RDatasetSpec::GetFileNameGlobs() const
{
   return fFileNameGlobs;
}

Long64_t RDatasetSpec::GetEntryRangeBegin() const
{
   return fEntryRange.fBegin;
}

Long64_t RDatasetSpec::GetEntryRangeEnd() const
{
   return fEntryRange.fEnd;
}

const ROOT::TreeUtils::RFriendInfo &RDatasetSpec::GetFriendInfo() const
{
   return fFriendInfo;
}

const std::vector<RDatasetSpec::RGroupInfo> &RDatasetSpec::GetGroupInfos() const
{
   return fGroupInfos;
}

RDatasetSpec::RGroupInfo::RGroupInfo(const std::string &name, Long64_t size, const RMetaData &metaData)
   : fName(name), fSize(size), fMetaData(metaData)
{
}

RDatasetSpec::RDatasetSpec(const std::vector<std::string> &trees, const std::vector<std::string> &fileGlobs,
                           const std::vector<RGroupInfo> &groupInfos, const ROOT::TreeUtils::RFriendInfo &friendInfo,
                           const REntryRange &entryRange)
   : fTreeNames(trees), fFileNameGlobs(fileGlobs), fEntryRange(entryRange), fFriendInfo(friendInfo),
     fGroupInfos(groupInfos)
{
}

RSpecBuilder &RSpecBuilder::AddGroup(const std::string &groupName, const std::string &treeName,
                                     const std::string &fileNameGlob, const RMetaData &metaData)
{
   // adding a single fileglob/tree, hence extend the vectors with 1 elem
   fTreeNames.reserve(fTreeNames.size() + 1);
   fTreeNames.emplace_back(treeName);
   fFileNameGlobs.reserve(fFileNameGlobs.size() + 1);
   fFileNameGlobs.emplace_back(fileNameGlob);

   // the group is of size 1, e.g. a single file glob
   fGroupInfos.reserve(fGroupInfos.size() + 1);
   fGroupInfos.emplace_back(RDatasetSpec::RGroupInfo(groupName, 1, metaData));
   return *this;
}

RSpecBuilder &RSpecBuilder::AddGroup(const std::string &groupName, const std::string &treeName,
                                     const std::vector<std::string> &fileNameGlobs, const RMetaData &metaData)
{
   // this constructor expects 1 tree name and multiple file names
   // however, in order to align many groups in TChain, we here copy the tree names multiple times
   // e.g. for N files, we store N (repeating) tree names to keep the alignment
   const auto nNewGlobs = fileNameGlobs.size();
   fTreeNames.reserve(fTreeNames.size() + nNewGlobs);
   for (auto i = 0u; i < nNewGlobs; ++i) // TODO: there might be a better intruction to do that
      fTreeNames.emplace_back(treeName);
   fFileNameGlobs.reserve(fFileNameGlobs.size() + nNewGlobs);
   fFileNameGlobs.insert(std::end(fFileNameGlobs), std::begin(fileNameGlobs), std::end(fileNameGlobs));

   fGroupInfos.reserve(fGroupInfos.size() + 1);
   fGroupInfos.emplace_back(RDatasetSpec::RGroupInfo(groupName, nNewGlobs, metaData));
   return *this;
}

RSpecBuilder &RSpecBuilder::AddGroup(const std::string &groupName,
                                     const std::vector<std::pair<std::string, std::string>> &treeAndFileNameGlobs,
                                     const RMetaData &metaData)
{
   const auto nNewGlobs = treeAndFileNameGlobs.size();
   fTreeNames.reserve(nNewGlobs);
   fFileNameGlobs.reserve(nNewGlobs);
   for (auto &p : treeAndFileNameGlobs) {
      fTreeNames.emplace_back(p.first);
      fFileNameGlobs.emplace_back(p.second);
   }

   fGroupInfos.reserve(fGroupInfos.size() + 1);
   fGroupInfos.emplace_back(RDatasetSpec::RGroupInfo(groupName, nNewGlobs, metaData));
   return *this;
}

RSpecBuilder &RSpecBuilder::AddGroup(const std::string &groupName, const std::vector<std::string> &trees,
                                     const std::vector<std::string> &files, const RMetaData &metaData)
{
   const auto nNewGlobs = files.size();
   if (trees.size() != 1 && trees.size() != nNewGlobs)
      throw std::logic_error("Mismatch between number of trees and file globs.");
   fTreeNames.reserve(fTreeNames.size() + nNewGlobs);
   fFileNameGlobs.reserve(fFileNameGlobs.size() + nNewGlobs);
   if (trees.size() == 1)
      for (auto i = 0u; i < nNewGlobs; ++i)
         fTreeNames.insert(std::end(fTreeNames), std::begin(trees), std::end(trees));
   else
      fTreeNames.insert(std::end(fTreeNames), std::begin(trees), std::end(trees));
   fFileNameGlobs.insert(std::end(fFileNameGlobs), std::begin(files), std::end(files));

   fGroupInfos.reserve(fGroupInfos.size() + 1);
   fGroupInfos.emplace_back(RDatasetSpec::RGroupInfo(groupName, nNewGlobs, metaData));
   return *this;
}

RSpecBuilder &
RSpecBuilder::WithFriends(const std::string &treeName, const std::string &fileNameGlob, const std::string &alias)
{
   fFriendInfo.AddFriend(treeName, fileNameGlob, alias);
   return *this;
}

RSpecBuilder &RSpecBuilder::WithFriends(const std::string &treeName, const std::vector<std::string> &fileNameGlobs,
                                        const std::string &alias)
{
   fFriendInfo.AddFriend(treeName, fileNameGlobs, alias);
   return *this;
}

RSpecBuilder &RSpecBuilder::WithFriends(const std::vector<std::pair<std::string, std::string>> &treeAndFileNameGlobs,
                                        const std::string &alias)
{
   fFriendInfo.AddFriend(treeAndFileNameGlobs, alias);
   return *this;
}

RSpecBuilder &RSpecBuilder::WithRange(const RDatasetSpec::REntryRange &entryRange)
{
   fEntryRange = entryRange;
   return *this;
}

RSpecBuilder &RSpecBuilder::WithFriends(const std::vector<std::string> &trees, const std::vector<std::string> &files,
                                        const std::string &alias)
{
   std::vector<std::pair<std::string, std::string>> target;
   target.reserve(files.size());
   std::transform(trees.begin(), trees.end(), files.begin(), std::back_inserter(target),
                  [](std::string a, std::string b) { return std::make_pair(a, b); });
   fFriendInfo.AddFriend(target, alias);
   return *this;
}

RDatasetSpec RSpecBuilder::Build()
{
   return std::move(RDatasetSpec(fTreeNames, fFileNameGlobs, fGroupInfos, fFriendInfo, fEntryRange));
}

} // namespace Experimental
} // namespace RDF
} // namespace ROOT
