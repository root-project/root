// Author: Ivan Kabadzhov CERN  11/2022

/*************************************************************************
 * Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RDF_RDATASETGROUP
#define ROOT_RDF_RDATASETGROUP

#include <ROOT/RDF/RMetaData.hxx>

#include <string>
#include <vector>

namespace ROOT {
namespace RDF {
namespace Experimental {

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

#endif // ROOT_RDF_RDATASETGROUP
