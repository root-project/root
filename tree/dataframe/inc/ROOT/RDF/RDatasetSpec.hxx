// Author: Vincenzo Eduardo Padulano CERN/UPV, Ivan Kabadzhov CERN  06/2022

/*************************************************************************
 * Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RDF_RDATASETSPEC
#define ROOT_RDF_RDATASETSPEC

#include <limits>
#include <string>
#include <utility> // std::pair
#include <vector>

#include <ROOT/RDF/RMetaData.hxx>
#include <ROOT/RFriendInfo.hxx>
#include <RtypesCore.h> // Long64_t

namespace ROOT {
namespace RDF {
namespace Experimental {

class RDatasetSpec {

public:
   struct REntryRange {
      Long64_t fBegin{0};
      Long64_t fEnd{std::numeric_limits<Long64_t>::max()};
      REntryRange();
      REntryRange(Long64_t endEntry);
      REntryRange(Long64_t startEntry, Long64_t endEntry);
   };

   // groups need to fulfill:
   // 1. preserve the original order -> arrange them in a vector, store also number of fileglobs in this group
   // 2. there is 1:1 correspondence between group and meta data => group and metadata can go together
   // Current solution: create a simple struct to hold the triplet {groupName, groupSize, MetaData}
   // This Group structure does not need an exposure to the user.
   struct RGroupInfo {
      std::string fName;   // name of the group
      Long64_t fSize;      // the matching between fileGlobs and group sizes is relative!
      RMetaData fMetaData; // behaves like a heterogenuous dictionary
      RGroupInfo(const std::string &name, Long64_t size, const RMetaData &metaData);
   };

private:
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
   REntryRange fEntryRange;                  ///< Start (inclusive) and end (exclusive) entry for the dataset processing
   ROOT::TreeUtils::RFriendInfo fFriendInfo; ///< List of friends
   std::vector<RGroupInfo> fGroupInfos;      ///< List of groups

public:
   RDatasetSpec(const std::vector<std::string> &trees, const std::vector<std::string> &fileGlobs,
                const std::vector<RGroupInfo> &groupInfos = {}, const ROOT::TreeUtils::RFriendInfo &friendInfo = {},
                const REntryRange &entryRange = {});

   const std::vector<std::string> &GetTreeNames() const;
   const std::vector<std::string> &GetFileNameGlobs() const;
   Long64_t GetEntryRangeBegin() const;
   Long64_t GetEntryRangeEnd() const;
   const ROOT::TreeUtils::RFriendInfo &GetFriendInfo() const;
   const std::vector<RGroupInfo> &GetGroupInfos() const;
};

class RSpecBuilder {
   std::vector<std::string> fTreeNames;
   std::vector<std::string> fFileNameGlobs;
   RDatasetSpec::REntryRange fEntryRange;             // global! range
   ROOT::TreeUtils::RFriendInfo fFriendInfo;          // global! friends
   std::vector<RDatasetSpec::RGroupInfo> fGroupInfos; // groups have relative order!

public:
   RSpecBuilder &AddGroup(const std::string &groupName, const std::string &treeName, const std::string &fileNameGlob,
                          const RMetaData &metaData);

   RSpecBuilder &AddGroup(const std::string &groupName, const std::string &treeName,
                          const std::vector<std::string> &fileNameGlobs, const RMetaData &metaData);

   RSpecBuilder &AddGroup(const std::string &groupName,
                          const std::vector<std::pair<std::string, std::string>> &treeAndFileNameGlobs,
                          const RMetaData &metaData);

   RSpecBuilder &AddGroup(const std::string &groupName, const std::vector<std::string> &trees,
                          const std::vector<std::string> &files, const RMetaData &metaData);

   RSpecBuilder &
   WithFriends(const std::string &treeName, const std::string &fileNameGlob, const std::string &alias = "");

   RSpecBuilder &WithFriends(const std::string &treeName, const std::vector<std::string> &fileNameGlobs,
                             const std::string &alias = "");

   RSpecBuilder &WithFriends(const std::vector<std::pair<std::string, std::string>> &treeAndFileNameGlobs,
                             const std::string &alias = "");

   RSpecBuilder &WithFriends(const std::vector<std::string> &trees, const std::vector<std::string> &files,
                             const std::string &alias = "");

   RSpecBuilder &WithRange(const RDatasetSpec::REntryRange &entryRange = {});

   RDatasetSpec Build();
};

} // namespace Experimental
} // namespace RDF
} // namespace ROOT

#endif // ROOT_RDF_RDATASETSPEC
