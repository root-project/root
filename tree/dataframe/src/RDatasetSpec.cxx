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

RDatasetSpec::REntryRange::REntryRange() {}

RDatasetSpec::REntryRange::REntryRange(Long64_t endEntry) : fEndEntry(endEntry) {}

RDatasetSpec::REntryRange::REntryRange(Long64_t startEntry, Long64_t endEntry)
   : fStartEntry(startEntry), fEndEntry(endEntry)
{
   if (fStartEntry > fEndEntry)
      throw std::logic_error("The starting entry cannot be larger than the ending entry in the "
                             "creation of a dataset specification.");
}

////////////////////////////////////////////////////////////////////////////
/// \brief Pass metadata specification to RDF
/// \param[in] treeName Name of the tree contained in the directory
/// \param[in] fileNameGlob Directories where the tree is stored
/// \param[in] entryRange  Struct containg {the first (inclusive), the last (exclusive)} global entry range
///
/// The filename glob supports the same type of expressions as TChain::Add()
RDatasetSpec::RDatasetSpec(const std::string &treeName, const std::string &fileNameGlob, const REntryRange &entryRange)
   : fTreeNames({treeName}), fFileNameGlobs({fileNameGlob}), fEntryRange(entryRange)
{
}

////////////////////////////////////////////////////////////////////////////
/// \brief Pass metadata specification to RDF
/// \param[in] treeName Name of the tree contained in the directory
/// \param[in] fileNameGlobs A vector of directories where the tree is stored
/// \param[in] entryRange  Struct containg {the first (inclusive), the last (exclusive)} global entry range
///
/// The filename glob supports the same type of expressions as TChain::Add()
RDatasetSpec::RDatasetSpec(const std::string &treeName, const std::vector<std::string> &fileNameGlobs,
                           const REntryRange &entryRange)
   : fTreeNames({treeName}), fFileNameGlobs(fileNameGlobs), fEntryRange(entryRange)
{
}

////////////////////////////////////////////////////////////////////////////
/// \brief Pass metadata specification to RDF
/// \param[in] treeAndFileNameGlobs A vector of pairs of tree names and their corresponding directories
/// \param[in] entryRange  Struct containg {the first (inclusive), the last (exclusive)} global entry range
///
/// The filename glob supports the same type of expressions as TChain::Add()
///
/// ### Example usage:
/// ~~~{.py}
/// spec = ROOT.RDF.RDatasetSpec([("tree1", "a.root"), ("tree2", "b.root")], (5, 10))
/// df = ROOT.RDataFrame(spec)
/// ~~~
RDatasetSpec::RDatasetSpec(const std::vector<std::pair<std::string, std::string>> &treeAndFileNameGlobs,
                           const REntryRange &entryRange)
   : fEntryRange(entryRange)
{
   fTreeNames.reserve(treeAndFileNameGlobs.size());
   fFileNameGlobs.reserve(treeAndFileNameGlobs.size());
   for (auto &p : treeAndFileNameGlobs) {
      fTreeNames.emplace_back(p.first);
      fFileNameGlobs.emplace_back(p.second);
   }
}

////////////////////////////////////////////////////////////////////////////
/// \brief Add a friend to the specification
/// \param[in] treeName Name of the tree contained in the directory
/// \param[in] fileNameGlob Directories where the tree is stored
/// \param[in] alias  String to refer to the particular friend
///
/// The filename glob supports the same type of expressions as TChain::Add()
void RDatasetSpec::AddFriend(const std::string &treeName, const std::string &fileNameGlob, const std::string &alias)
{
   fFriendInfo.fFriendNames.emplace_back(std::make_pair(treeName, alias));
   fFriendInfo.fFriendFileNames.emplace_back(std::vector<std::string>{fileNameGlob});
   fFriendInfo.fFriendChainSubNames.emplace_back(); // this is a tree
}

////////////////////////////////////////////////////////////////////////////
/// \brief Add a friend to the specification
/// \param[in] treeName Name of the tree contained in the directory
/// \param[in] fileNameGlobs A vector of directories where the tree is stored
/// \param[in] alias  String to refer to the particular friend
///
/// The filename glob supports the same type of expressions as TChain::Add()
void RDatasetSpec::AddFriend(const std::string &treeName, const std::vector<std::string> &fileNameGlobs,
                             const std::string &alias)
{
   fFriendInfo.fFriendNames.emplace_back(std::make_pair("", alias));
   fFriendInfo.fFriendFileNames.emplace_back(fileNameGlobs);
   fFriendInfo.fFriendChainSubNames.emplace_back(std::vector<std::string>(fileNameGlobs.size(), treeName));
}

////////////////////////////////////////////////////////////////////////////
/// \brief Add a friend to the specification
/// \param[in] treeAndFileNameGlobs A vector of pairs of tree names and their corresponding directories
/// \param[in] alias  String to refer to the particular friend
///
/// The filename glob supports the same type of expressions as TChain::Add()
///
/// ### Example usage:
/// ~~~{.py}
/// spec = ROOT.RDF.RDatasetSpec("tree", "file.root")
/// spec.AddFriend([("tree1", "a.root"), ("tree2", "b.root")], "alias")
/// df = ROOT.RDataFrame(spec)
/// ~~~
void RDatasetSpec::AddFriend(const std::vector<std::pair<std::string, std::string>> &treeAndFileNameGlobs,
                             const std::string &alias)
{
   fFriendInfo.fFriendNames.emplace_back(std::make_pair("", alias));
   fFriendInfo.fFriendFileNames.emplace_back();
   fFriendInfo.fFriendChainSubNames.emplace_back();
   // transform the vector of pairs to 2 vectors
   auto fSubNamesIt = std::back_inserter(fFriendInfo.fFriendChainSubNames.back());
   auto fNamesIt = std::back_inserter(fFriendInfo.fFriendFileNames.back());
   for (const auto &[subNames, names] : treeAndFileNameGlobs) {
      *fSubNamesIt = subNames;
      *fNamesIt = names;
   }
}

} // namespace RDF
} // namespace ROOT