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

/**
 * \class ROOT::RDF::Experimental::RDatasetSpec
 * \ingroup dataframe
 * \brief A dataset specification for RDataFrame.
 **/

////////////////////////////////////////////////////////////////////////////
/// \brief Construct an RDatasetSpec for one or more samples with the same tree name.
/// \param[in] treeName Name of the tree
/// \param[in] fileNameGlob Single file name or glob expression for the files where the tree(s) are stored
/// \param[in] entryRange The global entry range to be processed, {begin (inclusive), end (exclusive)}
///
/// The filename glob supports the same type of expressions as TChain::Add().
RDatasetSpec::RDatasetSpec(const std::string &treeName, const std::string &fileNameGlob, const REntryRange &entryRange)
   : fTreeNames({treeName}), fFileNameGlobs({fileNameGlob}), fEntryRange(entryRange)
{
}

////////////////////////////////////////////////////////////////////////////
/// \brief Construct an RDatasetSpec for one or more samples with the same tree name.
/// \param[in] treeName Name of the tree
/// \param[in] fileNameGlobs A vector of file names or glob expressions for the files where the trees are stored
/// \param[in] entryRange The global entry range to be processed, {begin (inclusive), end (exclusive)}
///
/// The filename glob supports the same type of expressions as TChain::Add().
RDatasetSpec::RDatasetSpec(const std::string &treeName, const std::vector<std::string> &fileNameGlobs,
                           const REntryRange &entryRange)
   : fTreeNames({treeName}), fFileNameGlobs(fileNameGlobs), fEntryRange(entryRange)
{
}

////////////////////////////////////////////////////////////////////////////
/// \brief Construct an RDatasetSpec for a chain of several trees (possibly having different names).
/// \param[in] treeAndFileNameGlobs A vector of pairs of tree names and their corresponding file names/globs
/// \param[in] entryRange The global entry range to be processed, {begin (inclusive), end (exclusive)}
///
/// The filename glob supports the same type of expressions as TChain::Add().
///
/// ### Example usage:
/// ~~~{.py}
/// spec = ROOT.RDF.Experimental.RDatasetSpec([("tree1", "a.root"), ("tree2", "b.root")], (5, 10))
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
/// \brief Add a friend tree or chain with the same tree name to the dataset specification.
/// \param[in] treeName Name of the tree
/// \param[in] fileNameGlob Single file name or glob expression for the files where the tree(s) are stored
/// \param[in] alias String to refer to the particular friend
///
/// The filename glob supports the same type of expressions as TChain::Add().
void RDatasetSpec::AddFriend(const std::string &treeName, const std::string &fileNameGlob, const std::string &alias)
{
   fFriendInfo.AddFriend(treeName, fileNameGlob, alias);
}

////////////////////////////////////////////////////////////////////////////
/// \brief Add a friend tree or chain with the same tree name to the dataset specification.
/// \param[in] treeName Name of the tree
/// \param[in] fileNameGlobs A vector of file names or glob expressions for the files where the trees are stored
/// \param[in] alias String to refer to the particular friend
///
/// The filename glob supports the same type of expressions as TChain::Add().
void RDatasetSpec::AddFriend(const std::string &treeName, const std::vector<std::string> &fileNameGlobs,
                             const std::string &alias)
{
   fFriendInfo.AddFriend(treeName, fileNameGlobs, alias);
}

////////////////////////////////////////////////////////////////////////////
/// \brief Add a friend tree or chain (possibly with different tree names) to the dataset specification.
/// \param[in] treeAndFileNameGlobs A vector of pairs of tree names and their corresponding file names/globs
/// \param[in] alias String to refer to the particular friend
///
/// The filename glob supports the same type of expressions as TChain::Add().
///
/// ### Example usage:
/// ~~~{.py}
/// spec = ROOT.RDF.Experimental.RDatasetSpec("tree", "file.root")
/// spec.AddFriend([("tree1", "a.root"), ("tree2", "b.root")], "alias")
/// df = ROOT.RDataFrame(spec)
/// ~~~
void RDatasetSpec::AddFriend(const std::vector<std::pair<std::string, std::string>> &treeAndFileNameGlobs,
                             const std::string &alias)
{
   fFriendInfo.AddFriend(treeAndFileNameGlobs, alias);
}

} // namespace Experimental
} // namespace RDF
} // namespace ROOT
