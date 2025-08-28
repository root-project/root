// Author: Vincenzo Eduardo Padulano CERN/UPV, Ivan Kabadzhov CERN  06/2022

/*************************************************************************
 * Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RDF/RDatasetSpec.hxx"
#include <ROOT/RFriendInfo.hxx>
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

////////////////////////////////////////////////////////////////////////////////
/// \brief Returns the collection of the dataset's sample names.
const std::vector<std::string> RDatasetSpec::GetSampleNames() const
{
   std::vector<std::string> sampleNames;
   sampleNames.reserve(fSamples.size());
   for (const auto &sample : fSamples)
      sampleNames.emplace_back(sample.GetSampleName());
   return sampleNames;
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Returns the collection of the dataset's tree names.
const std::vector<std::string> RDatasetSpec::GetTreeNames() const
{
   std::vector<std::string> treeNames;
   for (const auto &sample : fSamples) {
      const auto &trees = sample.GetTreeNames();
      treeNames.insert(std::end(treeNames), std::begin(trees), std::end(trees));
   }
   return treeNames;
}
////////////////////////////////////////////////////////////////////////////////
/// \brief Returns the collection of the dataset's paths to files, or globs if specified in input.
const std::vector<std::string> RDatasetSpec::GetFileNameGlobs() const
{
   std::vector<std::string> fileNames;
   for (const auto &sample : fSamples) {
      const auto &files = sample.GetFileNameGlobs();
      fileNames.insert(std::end(fileNames), std::begin(files), std::end(files));
   }
   return fileNames;
}
////////////////////////////////////////////////////////////////////////////////
/// \brief Returns the collection of the dataset's metadata (RMetaData class objects).
const std::vector<RMetaData> RDatasetSpec::GetMetaData() const
{
   std::vector<RMetaData> metaDatas;
   metaDatas.reserve(fSamples.size());
   for (const auto &sample : fSamples)
      metaDatas.emplace_back(sample.GetMetaData());
   return metaDatas;
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Returns the reference to the friend tree information.
const ROOT::TreeUtils::RFriendInfo &RDatasetSpec::GetFriendInfo() const
{
   return *std::any_cast<ROOT::TreeUtils::RFriendInfo>(&fFriendInfo);
}

ROOT::TreeUtils::RFriendInfo &RDatasetSpec::GetFriendInfo()
{
   return *std::any_cast<ROOT::TreeUtils::RFriendInfo>(&fFriendInfo);
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Returns the first entry as defined by the global range provided in the specification.
/// The first entry is inclusive.
Long64_t RDatasetSpec::GetEntryRangeBegin() const
{
   return fEntryRange.fBegin;
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Returns the last entry as defined by the global range provided in the specification.
/// The last entry is exclusive.
Long64_t RDatasetSpec::GetEntryRangeEnd() const
{
   return fEntryRange.fEnd;
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Returns a collection of instances of the RSample class.
/// RSample class represents a sample i.e. a grouping of trees (and their corresponding fileglobs) and, optionally, the
/// sample's metadata.
std::vector<RSample> RDatasetSpec::MoveOutSamples()
{
   return std::move(fSamples);
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Add sample (RSample class object) to the RDatasetSpec object.
/// \param[in] sample RSample class object.
/// RSample class represents a sample i.e. a grouping of trees (and their corresponding fileglobs) and, optionally, the
/// sample's metadata.
/// ### Example usage:
/// Our goal is to create an RDataFrame from the RDatasetSpec object.
/// In order to do that, we need to create an RSample object first.
/// In order to make this example even fuller, before we create the RSample object,
/// we also create the RMetaData object which will be associated with our RSample object.
/// Note that adding this metadata information to the RSample object is optional.
/// ~~~{.cpp}
/// // Create the RMetaData object which will be used to create the RSample object.
/// ROOT::RDF::Experimental::RMetaData meta;
/// meta.Add("sample_name", "name");
/// // Create the RSample object "mySample" with sample name "mySampleName", tree name "outputTree1",
/// // file name "outputFile.root" and associated metadata information.
/// ROOT::RDF::Experimental::RSample mySample("mySampleName", "outputTree1", "outputFile.root", meta);
/// // Create the RDatasetSpec object to which we add the sample (RSample object).
/// ROOT::RDF::Experimental::RDatasetSpec spec;
/// spec.AddSample(mySample);
/// // Finally, create an RDataFrame from the RDatasetSpec object.
/// auto df = ROOT::RDataFrame(spec);
/// ~~~
RDatasetSpec &RDatasetSpec::AddSample(RSample sample)
{
   sample.SetSampleId(fSamples.size());
   fSamples.push_back(std::move(sample));
   return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Add friend tree to RDatasetSpec object
/// \param[in] treeName Name of the tree.
/// \param[in] fileNameGlob Path to the file in which the tree is stored. Refer to TChain::Add for globbing rules.
/// \param[in] alias Alias for this friend.
///
/// ### Example usage:
/// ~~~{.cpp}
/// ROOT::RDF::Experimental::RSample s("mySample", "outputTree1", "outputFile.root");
/// ROOT::RDF::Experimental::RDatasetSpec spec;
/// spec.AddSample(s);
/// // Add friend tree "outputTree2" with the alias "myTreeFriend"
/// spec.WithGlobalFriends("outputTree2", "outputFile.root", "myTreeFriend");
/// auto df = ROOT::RDataFrame(spec);
/// ~~~
RDatasetSpec &
RDatasetSpec::WithGlobalFriends(const std::string &treeName, const std::string &fileNameGlob, const std::string &alias)
{
   GetFriendInfo().AddFriend(treeName, fileNameGlob, alias);
   return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Add friend tree to RDatasetSpec object
///
/// \param[in] treeName Name of the tree.
/// \param[in] fileNameGlobs Collection of paths to the files in which the tree is stored. Refer to TChain::Add for
/// globbing rules. \param[in] alias Alias for this friend.
RDatasetSpec &RDatasetSpec::WithGlobalFriends(const std::string &treeName,
                                              const std::vector<std::string> &fileNameGlobs, const std::string &alias)
{
   GetFriendInfo().AddFriend(treeName, fileNameGlobs, alias);
   return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Add friend tree to RDatasetSpec object
/// \param[in] treeAndFileNameGlobs Collection of pairs of paths to trees and paths to files.
/// \param[in] alias Alias for this friend.
RDatasetSpec &
RDatasetSpec::WithGlobalFriends(const std::vector<std::pair<std::string, std::string>> &treeAndFileNameGlobs,
                                const std::string &alias)
{
   GetFriendInfo().AddFriend(treeAndFileNameGlobs, alias);
   return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Add friend tree to RDatasetSpec object
/// \param[in] treeNames Collection of paths to trees.
/// \param[in] fileNameGlobs Collection of paths to files.
/// \param[in] alias Alias for this friend.
RDatasetSpec &RDatasetSpec::WithGlobalFriends(const std::vector<std::string> &treeNames,
                                        const std::vector<std::string> &fileNameGlobs, const std::string &alias)
{
   if (treeNames.size() != 1 && treeNames.size() != fileNameGlobs.size())
      throw std::logic_error("Mismatch between number of trees and file globs.");
   std::vector<std::pair<std::string, std::string>> target;
   target.reserve(fileNameGlobs.size());
   for (auto i = 0u; i < fileNameGlobs.size(); ++i)
      target.emplace_back(std::make_pair((treeNames.size() == 1u ? treeNames[0] : treeNames[i]), fileNameGlobs[i]));
   GetFriendInfo().AddFriend(target, alias);
   return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Create an RDatasetSpec object for a given range of entries
/// \param[in] entryRange
///
/// ## Example usage:
/// ~~~{.cpp}
/// ROOT::RDF::Experimental::RSample s("mySample", "outputTree1", "outputFile.root");
/// ROOT::RDF::Experimental::RDatasetSpec spec;
/// spec.AddSample(s);
/// // Set the entries range to be processed: including entry 1 and excluding entry 10.
/// spec.WithGlobalRange({1, 10});
/// auto df = ROOT::RDataFrame(spec);
/// ~~~
RDatasetSpec &RDatasetSpec::WithGlobalRange(const RDatasetSpec::REntryRange &entryRange)
{
   fEntryRange = entryRange;
   return *this;
}

} // namespace Experimental
} // namespace RDF
} // namespace ROOT

ROOT::RDF::Experimental::RDatasetSpec::RDatasetSpec() noexcept
{
   fFriendInfo = ROOT::TreeUtils::RFriendInfo{};
};
