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

#include <ROOT/RDF/RDatasetGroup.hxx>
#include <ROOT/RFriendInfo.hxx>
#include <RtypesCore.h> // Long64_t

namespace ROOT {
namespace RDF {
namespace Experimental {

/**
\class ROOT::RDF::Experimental::RDatasetSpec
\ingroup dataframe
\brief Class used to store semi-structured dataset specification

 This class is responsible for the creation of dataframe with metadata information.
*/
class RDatasetSpec {

   friend class RSpecBuilder;

public:
   struct REntryRange {
      Long64_t fBegin{0};
      Long64_t fEnd{std::numeric_limits<Long64_t>::max()};
      REntryRange();
      REntryRange(Long64_t endEntry);
      REntryRange(Long64_t startEntry, Long64_t endEntry);
   };

private:
   std::vector<RDatasetGroup> fDatasetGroups; ///< List of groups
   ROOT::TreeUtils::RFriendInfo fFriendInfo;  ///< List of friends
   REntryRange fEntryRange; ///< Start (inclusive) and end (exclusive) entry for the dataset processing

   RDatasetSpec(const std::vector<RDatasetGroup> &datasetGroups, const ROOT::TreeUtils::RFriendInfo &friendInfo = {},
                const REntryRange &entryRange = {});

public:
   const std::vector<std::string> GetGroupNames() const;
   const std::vector<std::string> GetTreeNames() const;
   const std::vector<std::string> GetFileNameGlobs() const;
   const std::vector<RMetaData> GetMetaData() const;
   const ROOT::TreeUtils::RFriendInfo &GetFriendInfo() const;
   Long64_t GetEntryRangeBegin() const;
   Long64_t GetEntryRangeEnd() const;

   /// \cond HIDDEN_SYMBOLS
   const std::vector<RDatasetGroup> &GetDatasetGroups() const;
   /// \endcond
};

class RSpecBuilder {
   std::vector<RDatasetGroup> fDatasetGroups; ///< List of groups
   ROOT::TreeUtils::RFriendInfo fFriendInfo;  ///< List of friends
   RDatasetSpec::REntryRange fEntryRange; ///< Start (inclusive) and end (exclusive) entry for the dataset processing

public:
   RSpecBuilder &AddGroup(RDatasetGroup datasetGroup);

   RSpecBuilder &
   WithFriends(const std::string &treeName, const std::string &fileNameGlob, const std::string &alias = "");

   RSpecBuilder &WithFriends(const std::string &treeName, const std::vector<std::string> &fileNameGlobs,
                             const std::string &alias = "");

   RSpecBuilder &WithFriends(const std::vector<std::pair<std::string, std::string>> &treeAndFileNameGlobs,
                             const std::string &alias = "");

   RSpecBuilder &WithFriends(const std::vector<std::string> &treeNames, const std::vector<std::string> &fileNameGlobs,
                             const std::string &alias = "");

   RSpecBuilder &WithRange(const RDatasetSpec::REntryRange &entryRange = {});

   RDatasetSpec Build();
};

} // namespace Experimental
} // namespace RDF
} // namespace ROOT

#endif // ROOT_RDF_RDATASETSPEC
