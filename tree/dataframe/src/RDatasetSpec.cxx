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

const std::vector<unsigned int> &RDatasetSpec::GetSizesOfFileGlobsBeforeExpansion() const
{
   return fSizesOfFileGlobsBeforeExpansion;
}

const std::vector<std::string> &RDatasetSpec::GetGroupNames() const
{
   return fGroupNames;
}

const std::vector<std::string> &RDatasetSpec::GetTreeNames() const
{
   return fTreeNames;
}

const std::vector<std::string> &RDatasetSpec::GetFileNameGlobs() const
{
   return fFileNameGlobs;
}

const std::vector<RMetaData> &RDatasetSpec::GetMetaDatas() const
{
   return fMetaDatas;
}

const ROOT::TreeUtils::RFriendInfo &RDatasetSpec::GetFriendInfo() const
{
   return fFriendInfo;
}

Long64_t RDatasetSpec::GetEntryRangeBegin() const
{
   return fEntryRange.fBegin;
}

Long64_t RDatasetSpec::GetEntryRangeEnd() const
{
   return fEntryRange.fEnd;
}

RDatasetSpec::RDatasetSpec(const std::vector<std::string> &groupNames, const std::vector<std::string> &treeNames,
                           const std::vector<std::string> &fileNameGlobs,
                           const std::vector<unsigned int> &sizesOfFileGlobsBeforeExpansion,
                           const std::vector<RMetaData> &metaDatas, const ROOT::TreeUtils::RFriendInfo &friendInfo,
                           const REntryRange &entryRange)
   : fGroupNames(groupNames), fTreeNames(treeNames), fFileNameGlobs(fileNameGlobs), fMetaDatas(metaDatas),
     fFriendInfo(friendInfo), fEntryRange(entryRange),
     fSizesOfFileGlobsBeforeExpansion(sizesOfFileGlobsBeforeExpansion)
{
}

RSpecBuilder &RSpecBuilder::AddGroup(const RDatasetGroup &datasetGroup)
{
   fGroupNames.reserve(fGroupNames.size() + 1);
   fGroupNames.emplace_back(datasetGroup.GetGroupName());

   const auto &currentTreeNames = datasetGroup.GetTreeNames();
   fTreeNames.reserve(fTreeNames.size() + currentTreeNames.size());
   fTreeNames.insert(std::end(fTreeNames), std::begin(currentTreeNames), std::end(currentTreeNames));

   const auto &currentFileNameGlobs = datasetGroup.GetFileNameGlobs();
   fFileNameGlobs.reserve(fFileNameGlobs.size() + currentFileNameGlobs.size());
   fFileNameGlobs.insert(std::end(fFileNameGlobs), std::begin(currentFileNameGlobs), std::end(currentFileNameGlobs));

   fMetaDatas.reserve(fMetaDatas.size() + 1);
   fMetaDatas.emplace_back(datasetGroup.GetMetaData());
   
   fSizesOfFileGlobsBeforeExpansion.reserve(fSizesOfFileGlobsBeforeExpansion.size() + 1);
   fSizesOfFileGlobsBeforeExpansion.emplace_back(currentFileNameGlobs.size());

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
   return RDatasetSpec(fGroupNames, fTreeNames, fFileNameGlobs, fSizesOfFileGlobsBeforeExpansion, fMetaDatas,
                       fFriendInfo, fEntryRange);
}

} // namespace Experimental
} // namespace RDF
} // namespace ROOT
