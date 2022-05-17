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
                        : throw std::logic_error("RDatasetSpec: fStartEntry cannot be larger than fEndEntry."))
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

   RDatasetSpec(const std::string &treeName, const std::string &fileName, REntryRange entryRange = {})
      : fTreeNames(std::vector<std::string>{treeName}), fFileNameGlobs(std::vector<std::string>{fileName}),
        fStartEntry(entryRange.fStartEntry), fEndEntry(entryRange.fEndEntry)
   {
   }

   RDatasetSpec(const std::string &treeName, const std::vector<std::string> &fileNames, REntryRange entryRange = {})
      : fTreeNames(std::vector<std::string>{treeName}), fFileNameGlobs(fileNames), fStartEntry(entryRange.fStartEntry),
        fEndEntry(entryRange.fEndEntry)
   {
   }

   RDatasetSpec(const std::vector<std::string> &treeNames, const std::vector<std::string> &fileNames,
                REntryRange entryRange = {})
      : fTreeNames(
           fileNames.size() != treeNames.size() && treeNames.size() != 1
              ? throw std::logic_error("RDatasetSpec exepcts either N trees and N files, or 1 tree and N files.")
              : treeNames),
        fFileNameGlobs(fileNames), fStartEntry(entryRange.fStartEntry), fEndEntry(entryRange.fEndEntry)
   {
   }

   void AddFriend(const std::string &treeName, const std::string &fileName)
   {
      fFriendInfo.fFriendNames.emplace_back(
         treeName, treeName); // TODO: user might specify alias, now it is the tree name by default
      fFriendInfo.fFriendFileNames.emplace_back(std::vector<std::string>{fileName});
      fFriendInfo.fFriendChainSubNames.emplace_back(std::vector<std::string>{}); // this is a tree
   }

   void AddFriend(const std::string &treeName, const std::vector<std::string> &fileNames)
   {
      fFriendInfo.fFriendNames.emplace_back(
         treeName, treeName); // TODO: user might specify alias, now it is the tree name by default
      fFriendInfo.fFriendFileNames.emplace_back(fileNames);
      fFriendInfo.fFriendChainSubNames.emplace_back(std::vector<std::string>{treeName}); // now this is a chain
   }

   void AddFriend(const std::vector<std::string> &treeNames, const std::vector<std::string> &fileNames)
   {
      if (fileNames.size() != treeNames.size() && treeNames.size() != 1)
         throw std::logic_error("RDatasetSpec's friend exepcts either N trees and N files, or 1 tree and N files.");
      fFriendInfo.fFriendNames.emplace_back(
         treeNames[0], treeNames[0]); // TODO: user might specify alias, now it is the FIRST tree name by default
      fFriendInfo.fFriendFileNames.emplace_back(fileNames);
      fFriendInfo.fFriendChainSubNames.emplace_back(treeNames); // now this is a chain
   }
};

} // namespace RDF
} // namespace ROOT

#endif // ROOT_RDF_RDATASETSPEC
