// Author: Ivan Kabadzhov CERN  10/2022

/*************************************************************************
 * Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RDF_RMETADATA
#define ROOT_RDF_RMETADATA

#include <nlohmann/json.hpp>

#include <string>

namespace ROOT {
namespace RDF {
namespace Experimental {

/**
\class ROOT::RDF::Experimental::RMetaData
\ingroup dataframe
\brief Class behaving as a heterogenuous dictionary to store dataset metadata
 
 This class should be passed to an RDatasetGroup object which represents a single dataset group.
 Once a dataframe is built with RMetaData object, it could be accessed via DefinePerSample.
*/
class RMetaData {
   nlohmann::json fJson;

public:
   void Add(const std::string &key, int val);
   void Add(const std::string &key, double val);
   void Add(const std::string &key, const std::string &val);

   const std::string Dump(const std::string &key) const; // always returns a string
   int GetI(const std::string &key) const;
   double GetD(const std::string &key) const;
   const std::string GetS(const std::string &key) const;
   int GetI(const std::string &key, int defaultVal) const;
   double GetD(const std::string &key, double defaultVal) const;
   const std::string GetS(const std::string &key, std::string defaultVal) const;
};

/**
\class ROOT::RDF::Experimental::RDatasetGroup
\ingroup dataframe
\brief Class representing a dataset group (mapping of trees (and their fileglobs) to metadata)
 
 This class should be passed to RSpecBuilder in order to build a RDataFrame.
*/
class RDatasetGroup {
   std::string fGroupName;
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
   RMetaData fMetaData;

   unsigned int fGroupId{0}; // global group index, set inside of the RDatasetSpec

public:
   RDatasetGroup(const std::string &groupName, const std::string &treeName, const std::string &fileNameGlob,
                 const RMetaData &metaData = {});

   RDatasetGroup(const std::string &groupName, const std::string &treeName,
                 const std::vector<std::string> &fileNameGlobs, const RMetaData &metaData = {});

   RDatasetGroup(const std::string &groupName,
                 const std::vector<std::pair<std::string, std::string>> &treeAndFileNameGlobs,
                 const RMetaData &metaData = {});

   RDatasetGroup(const std::string &groupName, const std::vector<std::string> &treeNames,
                 const std::vector<std::string> &fileNameGlobs, const RMetaData &metaData = {});

   /// \cond HIDDEN_SYMBOLS
   RDatasetGroup() {} // empty constructor to make RSampleInfo happy
   /// \endcond

   const std::string &GetGroupName() const;
   const std::vector<std::string> &GetTreeNames() const;
   const std::vector<std::string> &GetFileNameGlobs() const;
   const RMetaData &GetMetaData() const;

   /// \cond HIDDEN_SYMBOLS
   unsigned int GetGroupId() const; // intended to be used only after the RDataSpec is build, otherwise is 0
   void SetGroupId(unsigned int id);
   /// \endcond
   };

} // namespace Experimental
} // namespace RDF
} // namespace ROOT

#endif // ROOT_RDF_RMETADATA
