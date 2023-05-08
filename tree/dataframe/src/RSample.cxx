// Author: Ivan Kabadzhov CERN  11/2022

/*************************************************************************
 * Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RDF/RSample.hxx"
#include "TChain.h"
#include <stdexcept> // std::logic_error

namespace ROOT {
namespace RDF {
namespace Experimental {

RSample::RSample(const std::string &sampleName, const std::string &treeName, const std::string &fileNameGlob,
                 const RMetaData &metaData)
   : RSample(sampleName, std::vector<std::string>{treeName}, std::vector<std::string>{fileNameGlob}, metaData)
{
}

RSample::RSample(const std::string &sampleName, const std::string &treeName,
                 const std::vector<std::string> &fileNameGlobs, const RMetaData &metaData)
   : RSample(sampleName, std::vector<std::string>(fileNameGlobs.size(), treeName), fileNameGlobs, metaData)
{
}

RSample::RSample(const std::string &sampleName,
                 const std::vector<std::pair<std::string, std::string>> &treeAndFileNameGlobs,
                 const RMetaData &metaData)
   : fSampleName(sampleName), fMetaData(metaData)
{
   // avoid constructing and destructing the helper TChain here if we don't need to
   if (treeAndFileNameGlobs.empty())
      return;

   TChain chain;
   for (const auto &p : treeAndFileNameGlobs) {
      const auto fullpath = p.second + "?#" + p.first;
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

RSample::RSample(const std::string &sampleName, const std::vector<std::string> &treeNames,
                 const std::vector<std::string> &fileNameGlobs, const RMetaData &metaData)
   : fSampleName(sampleName), fMetaData(metaData)
{
   if (treeNames.size() != 1 && treeNames.size() != fileNameGlobs.size())
      throw std::logic_error("Mismatch between number of trees and file globs.");
   TChain chain;
   for (auto i = 0u; i < fileNameGlobs.size(); ++i) {
      const auto fullpath = fileNameGlobs[i] + "?#" + (treeNames.size() == 1u ? treeNames[0] : treeNames[i]);
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

const std::string &RSample::GetSampleName() const
{
   return fSampleName;
}

const std::vector<std::string> &RSample::GetTreeNames() const
{
   return fTreeNames;
}

const std::vector<std::string> &RSample::GetFileNameGlobs() const
{
   return fFileNameGlobs;
}

const RMetaData &RSample::GetMetaData() const
{
   return fMetaData;
}

unsigned int RSample::GetSampleId() const
{
   return fSampleId;
}

void RSample::SetSampleId(unsigned int id)
{
   fSampleId = id;
}

} // namespace Experimental
} // namespace RDF
} // namespace ROOT
