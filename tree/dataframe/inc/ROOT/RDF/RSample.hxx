// Author: Ivan Kabadzhov CERN  11/2022

/*************************************************************************
 * Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RDF_RSAMPLE
#define ROOT_RDF_RSAMPLE

#include <ROOT/RDF/RMetaData.hxx>

#include <string>
#include <vector>

namespace ROOT {
namespace RDF {
namespace Experimental {

/**
\ingroup dataframe
\brief Class representing a sample which is a grouping of trees and their fileglobs, and, optionally, the sample's
metadata information via the RMetaData object.

 The class is passed to an RDatasetSpec object in order to build an RDataFrame.

 For example, an RSample object can be built as follows:
 ~~~{.cpp}
 // First, create the RMetaData object (to, optionally, add to the sample)
 ROOT::RDF::Experimental::RMetaData meta;
 meta.Add("sample_name", "name"");
 // Create an RSample with metadata information
 ROOT::RDF::Experimental::RSample mySample("mySampleName", "outputTree1", "outputFile.root", meta);
 ~~~
*/
class RSample {
   /// Name of the sample.
   std::string fSampleName;
   /**
    * A list of names of trees.
    * This list should go in lockstep with fFileNameGlobs, only in case this dataset is a TChain where each file
    * contains its own tree with a different name from the global name of the dataset.
    * Otherwise, fTreeNames contains 1 treename, that is common for all file globs.
    */
   std::vector<std::string> fTreeNames;
   /**
    * A list of file names.
    * They can contain the globbing characters supported by TChain. See TChain::Add for more information.
    */
   std::vector<std::string> fFileNameGlobs;
   /// An instance of the RMetaData class.
   RMetaData fMetaData;

   /// Global sample index, set inside of the RDatasetSpec.
   unsigned int fSampleId{0};

public:
   RSample(RSample &&) = default;
   RSample &operator=(RSample &&) = default;
   RSample(const RSample &) = default;
   RSample &operator=(const RSample &) = default;
   RSample() = delete;

   RSample(const std::string &sampleName, const std::string &treeName, const std::string &fileNameGlob,
           const RMetaData &metaData = {});

   RSample(const std::string &sampleName, const std::string &treeName, const std::vector<std::string> &fileNameGlobs,
           const RMetaData &metaData = {});

   RSample(const std::string &sampleName, const std::vector<std::pair<std::string, std::string>> &treeAndFileNameGlobs,
           const RMetaData &metaData = {});

   RSample(const std::string &sampleName, const std::vector<std::string> &treeNames,
           const std::vector<std::string> &fileNameGlobs, const RMetaData &metaData = {});

   const std::string &GetSampleName() const;
   const std::vector<std::string> &GetTreeNames() const;
   const std::vector<std::string> &GetFileNameGlobs() const;
   const RMetaData &GetMetaData() const;

   /// \cond HIDDEN_SYMBOLS
   unsigned int GetSampleId() const; // intended to be used only after the RDataSpec is build, otherwise is 0
   void SetSampleId(unsigned int id);
   /// \endcond
};

} // namespace Experimental
} // namespace RDF
} // namespace ROOT

#endif // ROOT_RDF_RSAMPLE
