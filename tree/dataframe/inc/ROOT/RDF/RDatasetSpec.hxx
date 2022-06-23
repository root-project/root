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

#include <string>
#include <vector>
#include <limits>
#include <stdexcept> // std::logic_error
#include <algorithm> // std::transform

#include <RtypesCore.h>
#include <ROOT/InternalTreeUtils.hxx>

namespace ROOT {

namespace RDF {

struct RDatasetSpec {

   struct REntryRange {
      Long64_t fStartEntry{0};
      Long64_t fEndEntry{std::numeric_limits<Long64_t>::max()};
      REntryRange() {}
      REntryRange(Long64_t endEntry) : fEndEntry(endEntry) {}
      REntryRange(Long64_t startEntry, Long64_t endEntry)
         : fStartEntry(startEntry),
           fEndEntry(endEntry >= startEntry
                        ? endEntry
                        : throw std::logic_error("The starting entry cannot be larger than the ending entry in the "
                                                 "creation of a dataset specification."))
      {
      }
   };

   /**
    * A list of names of trees.
    * This list should go in lockstep with fFileNameGlobs, only in case this dataset is a TChain where each file
    * contains its own tree with a different name from the global name of the dataset.
    */
   std::vector<std::string> fTreeNames{};

   /**
    * A list of file names.
    * They can contain the globbing characters supported by TChain. See TChain::Add for more information.
    */
   std::vector<std::string> fFileNameGlobs{};

   Long64_t fStartEntry{}; ///< The entry where the dataset processing should start (inclusive).
   Long64_t fEndEntry{};   ///< The entry where the dataset processing should end (exclusive).

   ROOT::Internal::TreeUtils::RFriendInfo fFriendInfo{}; ///< List of friends

   RDatasetSpec(const std::string &treeName, const std::string &fileNameGlob, const REntryRange &entryRange = {});

   RDatasetSpec(const std::string &treeName, const std::vector<std::string> &fileNameGlobs,
                const REntryRange &entryRange = {});

   RDatasetSpec(const std::vector<std::pair<std::string, std::string>> &treeAndFileNameGlobs,
                const REntryRange &entryRange = {});

   void AddFriend(const std::string &treeName, const std::string &fileNameGlob, const std::string &alias = "");

   void
   AddFriend(const std::string &treeName, const std::vector<std::string> &fileNameGlobs, const std::string &alias = "");

   void AddFriend(const std::vector<std::pair<std::string, std::string>> &treeAndFileNameGlobs,
                  const std::string &alias = "");
};

} // namespace RDF
} // namespace ROOT

#endif // ROOT_RDF_RDATASETSPEC
