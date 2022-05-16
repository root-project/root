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

   std::string fDatasetName{}; ///< The name of the dataset to process.
   /**
    * A list of file names.
    * They can contain the globbing characters supported by TChain. See TChain::Add for more information.
    */
   std::vector<std::string> fFileNameGlobs{};
   ULong64_t fStartEntry{}; ///< The entry where the dataset processing should start (inclusive).
   ULong64_t fEndEntry{};   ///< The entry where the dataset processing should end (exclusive).

   /**
    * A list of names of trees.
    * This list should go in lockstep with fFileNameGlobs, only in case this dataset is a TChain where each file
    * contains its own tree with a different name from the global name of the dataset.
    */
   std::vector<std::string> fSubTreeNames{};

   RDatasetSpec(const std::string &datasetName, const std::string &fileName, REntryRange entryRange = {},
                const std::vector<std::string> &subTreenames = {})
      : fDatasetName(datasetName), fFileNameGlobs(std::vector<std::string>{fileName}),
        fStartEntry(entryRange.fStartEntry), fEndEntry(entryRange.fEndEntry), fSubTreeNames(subTreenames)
   {
   }

   RDatasetSpec(const std::string &datasetName, const std::vector<std::string> &fileNames, REntryRange entryRange = {},
                const std::vector<std::string> &subTreenames = {})
      : fDatasetName(datasetName), fFileNameGlobs(fileNames), fStartEntry(entryRange.fStartEntry),
        fEndEntry(entryRange.fEndEntry), fSubTreeNames(subTreenames)
   {
   }
};

} // namespace RDF
} // namespace ROOT

#endif // ROOT_RDF_RDATASETSPEC
