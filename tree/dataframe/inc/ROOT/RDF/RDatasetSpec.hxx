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

#include <ROOT/RDF/RSample.hxx>
#include <ROOT/RFriendInfo.hxx>
#include <RtypesCore.h> // Long64_t

namespace ROOT {
namespace Detail {
namespace RDF {
class RLoopManager;
}
} // namespace Detail
namespace RDF {
namespace Experimental {

/**
\ingroup dataframe
\brief A dataset specification for RDataFrame.
*/
class RDatasetSpec {
   friend class ::ROOT::Detail::RDF::RLoopManager; // for MoveOutSamples

public:
   struct REntryRange {
      Long64_t fBegin{0};
      Long64_t fEnd{std::numeric_limits<Long64_t>::max()};
      REntryRange();
      REntryRange(Long64_t endEntry);
      REntryRange(Long64_t startEntry, Long64_t endEntry);
   };

private:
   std::vector<RSample> fSamples;             ///< List of samples
   ROOT::TreeUtils::RFriendInfo fFriendInfo;  ///< List of friends
   REntryRange fEntryRange; ///< Start (inclusive) and end (exclusive) entry for the dataset processing

   std::vector<RSample> MoveOutSamples();

public:
   RDatasetSpec() = default;

   const std::vector<std::string> GetSampleNames() const;
   const std::vector<std::string> GetTreeNames() const;
   const std::vector<std::string> GetFileNameGlobs() const;
   const std::vector<RMetaData> GetMetaData() const;
   const ROOT::TreeUtils::RFriendInfo &GetFriendInfo() const;
   Long64_t GetEntryRangeBegin() const;
   Long64_t GetEntryRangeEnd() const;

   RDatasetSpec &AddSample(RSample sample);

   RDatasetSpec &
   WithGlobalFriends(const std::string &treeName, const std::string &fileNameGlob, const std::string &alias = "");

   RDatasetSpec &WithGlobalFriends(const std::string &treeName, const std::vector<std::string> &fileNameGlobs,
                             const std::string &alias = "");

   RDatasetSpec &WithGlobalFriends(const std::vector<std::pair<std::string, std::string>> &treeAndFileNameGlobs,
                             const std::string &alias = "");

   RDatasetSpec &WithGlobalFriends(const std::vector<std::string> &treeNames,
                                   const std::vector<std::string> &fileNameGlobs, const std::string &alias = "");

   RDatasetSpec &WithGlobalRange(const RDatasetSpec::REntryRange &entryRange = {});
};

} // namespace Experimental
} // namespace RDF
} // namespace ROOT

#endif // ROOT_RDF_RDATASETSPEC
