// Author: Vincenzo Eduardo Padulano CERN/UPV 05/2022

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
      ULong64_t fStartEntry{0};
      ULong64_t fEndEntry{std::numeric_limits<ULong64_t>::max()};
      REntryRange() {}
      REntryRange(ULong64_t endEntry) : fEndEntry(endEntry) {}
      REntryRange(ULong64_t startEntry, ULong64_t endEntry)
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

   ULong64_t fStartEntry{}; ///< The entry where the dataset processing should start (inclusive).
   ULong64_t fEndEntry{};   ///< The entry where the dataset processing should end (exclusive).

   ROOT::Internal::TreeUtils::RFriendInfo fFriendInfo{}; ///< List of friends

   RDatasetSpec(const std::string &treeName, const std::string &fileName, const REntryRange &entryRange = {})
      : fTreeNames(std::vector<std::string>{std::move(treeName)}),
        fFileNameGlobs(std::vector<std::string>{std::move(fileName)}), fStartEntry(entryRange.fStartEntry),
        fEndEntry(entryRange.fEndEntry)
   {
   }

   RDatasetSpec(const std::string &treeName, const std::vector<std::string> &fileNames,
                const REntryRange &entryRange = {})
      : fTreeNames(std::vector<std::string>{std::move(treeName)}), fFileNameGlobs(std::move(fileNames)),
        fStartEntry(entryRange.fStartEntry), fEndEntry(entryRange.fEndEntry)
   {
   }

   RDatasetSpec(const std::vector<std::pair<std::string, std::string>> &treeAndFileNames,
                const REntryRange &entryRange = {})
      : fStartEntry(entryRange.fStartEntry), fEndEntry(entryRange.fEndEntry)
   {
      fTreeNames.reserve(treeAndFileNames.size());
      fFileNameGlobs.reserve(treeAndFileNames.size());
      for (auto &p : treeAndFileNames) {
         fTreeNames.emplace_back(std::move(p.first));
         fFileNameGlobs.emplace_back(std::move(p.second));
      }
   }

   void AddFriend(const std::string &treeName, const std::string &fileName, const std::string &alias = "")
   {
      fFriendInfo.fFriendNames.emplace_back(std::make_pair(std::move(treeName), std::move(alias)));
      fFriendInfo.fFriendFileNames.emplace_back(std::vector<std::string>{std::move(fileName)});
      fFriendInfo.fFriendChainSubNames.emplace_back(); // this is a tree
   }

   void AddFriend(const std::string &treeName, const std::vector<std::string> &fileNames, const std::string &alias = "")
   {
      fFriendInfo.fFriendNames.emplace_back(std::make_pair("", std::move(alias)));
      fFriendInfo.fFriendFileNames.emplace_back(std::move(fileNames));
      fFriendInfo.fFriendChainSubNames.emplace_back(std::vector<std::string>(fileNames.size(), std::move(treeName)));
   }

   void
   AddFriend(const std::vector<std::pair<std::string, std::string>> &treeAndFileNames, const std::string &alias = "")
   {
      fFriendInfo.fFriendNames.emplace_back(std::make_pair("", std::move(alias)));
      fFriendInfo.fFriendFileNames.emplace_back();
      fFriendInfo.fFriendChainSubNames.emplace_back();
      auto &fileNames = fFriendInfo.fFriendFileNames.back();
      auto &chainSubNames = fFriendInfo.fFriendChainSubNames.back();
      fileNames.reserve(treeAndFileNames.size());
      chainSubNames.reserve(treeAndFileNames.size());
      for (auto &p : treeAndFileNames) {
         chainSubNames.emplace_back(std::move(p.first));
         fileNames.emplace_back(std::move(p.second));
      }
   }
};

} // namespace RDF
} // namespace ROOT

#endif // ROOT_RDF_RDATASETSPEC
