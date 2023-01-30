// Author: Ivan Kabadzhov CERN  11/2022

/*************************************************************************
 * Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RDF/RDatasetGroup.hxx"
#include "TChain.h"
#include <stdexcept> // std::logic_error

namespace ROOT {
namespace RDF {
namespace Experimental {

RDatasetGroup::RDatasetGroup(const std::string &groupName, const std::string &treeName, const std::string &fileNameGlob,
                             const RMetaData &metaData)
   : RDatasetGroup(groupName, std::vector<std::string>{treeName}, std::vector<std::string>{fileNameGlob}, metaData)
{
}

RDatasetGroup::RDatasetGroup(const std::string &groupName, const std::string &treeName,
                             const std::vector<std::string> &fileNameGlobs, const RMetaData &metaData)
   : RDatasetGroup(groupName, std::vector<std::string>(fileNameGlobs.size(), treeName), fileNameGlobs, metaData)
{
}

RDatasetGroup::RDatasetGroup(const std::string &groupName,
                             const std::vector<std::pair<std::string, std::string>> &treeAndFileNameGlobs,
                             const RMetaData &metaData)
   : fGroupName(groupName), fMetaData(metaData)
{
   // avoid constructing and destructing the helper TChain here if we don't need to
   if (treeAndFileNameGlobs.empty())
      return;

   TChain chain;
   for (const auto &p : treeAndFileNameGlobs) {
      const auto fullpath = p.second + "/" + p.first;
      chain.Add(fullpath.c_str());
   }
   const auto &expandedNames = chain.GetListOfFiles();
   fTreeNames.reserve(expandedNames->GetEntries());
   fFileNameGlobs.reserve(expandedNames->GetEntries());
   for (auto i = 0; i < expandedNames->GetEntries(); ++i) {
      fTreeNames.emplace_back(expandedNames->At(i)->GetName());
      fFileNameGlobs.emplace_back(expandedNames->At(i)->GetTitle());
   }
}

RDatasetGroup::RDatasetGroup(const std::string &groupName, const std::vector<std::string> &treeNames,
                             const std::vector<std::string> &fileNameGlobs, const RMetaData &metaData)
   : fGroupName(groupName), fMetaData(metaData)
{
   if (treeNames.size() != 1 && treeNames.size() != fileNameGlobs.size())
      throw std::logic_error("Mismatch between number of trees and file globs.");
   TChain chain;
   for (auto i = 0u; i < fileNameGlobs.size(); ++i) {
      const auto fullpath = fileNameGlobs[i] + "/" + (treeNames.size() == 1u ? treeNames[0] : treeNames[i]);
      chain.Add(fullpath.c_str());
   }
   const auto &expandedNames = chain.GetListOfFiles();
   fTreeNames.reserve(expandedNames->GetEntries());
   fFileNameGlobs.reserve(expandedNames->GetEntries());
   for (auto i = 0; i < expandedNames->GetEntries(); ++i) {
      fTreeNames.emplace_back(expandedNames->At(i)->GetName());
      fFileNameGlobs.emplace_back(expandedNames->At(i)->GetTitle());
   }
}

const std::string &RDatasetGroup::GetGroupName() const
{
   return fGroupName;
}

const std::vector<std::string> &RDatasetGroup::GetTreeNames() const
{
   return fTreeNames;
}

const std::vector<std::string> &RDatasetGroup::GetFileNameGlobs() const
{
   return fFileNameGlobs;
}

const RMetaData &RDatasetGroup::GetMetaData() const
{
   return fMetaData;
}

unsigned int RDatasetGroup::GetGroupId() const
{
   return fGroupId;
}

void RDatasetGroup::SetGroupId(unsigned int id)
{
   fGroupId = id;
}

} // namespace Experimental
} // namespace RDF
} // namespace ROOT
