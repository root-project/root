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

#include <any>
#include <limits>
#include <string>
#include <utility> // std::pair
#include <vector>

#include <ROOT/RDF/RSample.hxx>
#include <RtypesCore.h> // Long64_t

namespace ROOT::TreeUtils {
struct RFriendInfo;
}

namespace ROOT {
namespace Detail {
namespace RDF {
class RLoopManager;
} // namespace RDF
} // namespace Detail

namespace RDF {
namespace Experimental {
class RDatasetSpec;
class RSample;
} // namespace Experimental
} // namespace RDF

namespace Internal {
namespace RDF {
std::vector<ROOT::RDF::Experimental::RSample> MoveOutSamples(ROOT::RDF::Experimental::RDatasetSpec &spec);
}
} // namespace Internal

namespace RDF {
namespace Experimental {

// clang-format off
/**
\class ROOT::RDF::Experimental::RDatasetSpec
\ingroup dataframe
\brief The dataset specification for RDataFrame.

This class allows users to create the dataset specification for RDataFrame 
to which they add samples (using the RSample class object) with tree names and file names, 
and, optionally, the metadata information (using the RMetaData class objects). 
Adding global friend trees and/or setting the range of events to be processed
are also available.

Note, there exists yet another method to build RDataFrame from the dataset information using the JSON file format: \ref FromSpec(const std::string &jsonFile) "FromSpec()". 
*/

class RDatasetSpec {
   // clang-format on 
   friend class ::ROOT::Detail::RDF::RLoopManager; // for MoveOutSamples
   friend std::vector<ROOT::RDF::Experimental::RSample> ROOT::Internal::RDF::MoveOutSamples(ROOT::RDF::Experimental::RDatasetSpec &); 


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
   std::any fFriendInfo;  ///< List of friends
   REntryRange fEntryRange; ///< Start (inclusive) and end (exclusive) entry for the dataset processing
   std::vector<RSample> MoveOutSamples();
   ROOT::TreeUtils::RFriendInfo &GetFriendInfo();

public:
   RDatasetSpec() noexcept;

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
