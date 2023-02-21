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

const std::vector<std::string> RDatasetSpec::GetSampleNames() const
{
   std::vector<std::string> sampleNames;
   sampleNames.reserve(fSamples.size());
   for (const auto &sample : fSamples)
      sampleNames.emplace_back(sample.GetSampleName());
   return sampleNames;
}

const std::vector<std::string> RDatasetSpec::GetTreeNames() const
{
   std::vector<std::string> treeNames;
   for (const auto &sample : fSamples) {
      const auto &trees = sample.GetTreeNames();
      treeNames.insert(std::end(treeNames), std::begin(trees), std::end(trees));
   }
   return treeNames;
}

const std::vector<std::string> RDatasetSpec::GetFileNameGlobs() const
{
   std::vector<std::string> fileNames;
   for (const auto &sample : fSamples) {
      const auto &files = sample.GetFileNameGlobs();
      fileNames.insert(std::end(fileNames), std::begin(files), std::end(files));
   }
   return fileNames;
}

const std::vector<RMetaData> RDatasetSpec::GetMetaData() const
{
   std::vector<RMetaData> metaDatas;
   metaDatas.reserve(fSamples.size());
   for (const auto &sample : fSamples)
      metaDatas.emplace_back(sample.GetMetaData());
   return metaDatas;
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

std::vector<RSample> RDatasetSpec::MoveOutSamples()
{
   return std::move(fSamples);
}

RDatasetSpec &RDatasetSpec::AddSample(RSample sample)
{
   sample.SetSampleId(fSamples.size());
   fSamples.push_back(std::move(sample));
   return *this;
}

RDatasetSpec &
RDatasetSpec::WithGlobalFriends(const std::string &treeName, const std::string &fileNameGlob, const std::string &alias)
{
   fFriendInfo.AddFriend(treeName, fileNameGlob, alias);
   return *this;
}

RDatasetSpec &RDatasetSpec::WithGlobalFriends(const std::string &treeName,
                                              const std::vector<std::string> &fileNameGlobs, const std::string &alias)
{
   fFriendInfo.AddFriend(treeName, fileNameGlobs, alias);
   return *this;
}

RDatasetSpec &
RDatasetSpec::WithGlobalFriends(const std::vector<std::pair<std::string, std::string>> &treeAndFileNameGlobs,
                                const std::string &alias)
{
   fFriendInfo.AddFriend(treeAndFileNameGlobs, alias);
   return *this;
}

RDatasetSpec &RDatasetSpec::WithGlobalRange(const RDatasetSpec::REntryRange &entryRange)
{
   fEntryRange = entryRange;
   return *this;
}

RDatasetSpec &RDatasetSpec::WithGlobalFriends(const std::vector<std::string> &treeNames,
                                        const std::vector<std::string> &fileNameGlobs, const std::string &alias)
{
   if (treeNames.size() != 1 && treeNames.size() != fileNameGlobs.size())
      throw std::logic_error("Mismatch between number of trees and file globs.");
   std::vector<std::pair<std::string, std::string>> target;
   target.reserve(fileNameGlobs.size());
   for (auto i = 0u; i < fileNameGlobs.size(); ++i)
      target.emplace_back(std::make_pair((treeNames.size() == 1u ? treeNames[0] : treeNames[i]), fileNameGlobs[i]));
   fFriendInfo.AddFriend(target, alias);
   return *this;
}

} // namespace Experimental
} // namespace RDF
} // namespace ROOT
