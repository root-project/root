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

#include <RtypesCore.h>

namespace ROOT {

namespace RDF {

struct RDatasetSpec {
   std::string fDatasetName{}; ///< The name of the dataset to process.
   /**
    * A list of file names.
    * They can contain the globbing characters supported by TChain. See TChain::Add for more information.
    */
   std::vector<std::string> fFileNameGlobs{};
   ULong64_t fStartEntry{};                    ///< The entry where the dataset processing should start (inclusive).
   ULong64_t fEndEntry{};                      ///< The entry where the dataset processing should end (exclusive).
   std::vector<std::string> fDefaultColumns{}; ///< A list of column names to process in the dataset.
   /**
    * A list of names of trees.
    * This list should go in lockstep with fFileNameGlobs, only in case this dataset is a TChain where each file
    * contains its own tree with a different name from the global name of the dataset.
    */
   std::vector<std::string> fSubTreeNames{};

   RDatasetSpec(const std::string &datasetName, const std::string &fileName, ULong64_t startEntry = 0,
                ULong64_t endEntry = 0, const std::vector<std::string> &defaultColumns = {},
                const std::vector<std::string> &subTreenames = {})
      : fDatasetName(datasetName), fFileNameGlobs(std::vector<std::string>{fileName}), fStartEntry(startEntry),
        fEndEntry(endEntry), fDefaultColumns(defaultColumns), fSubTreeNames(subTreenames)
   {
   }

   RDatasetSpec(const std::string &datasetName, const std::vector<std::string> &fileNames, ULong64_t startEntry = 0,
                ULong64_t endEntry = 0, const std::vector<std::string> &defaultColumns = {},
                const std::vector<std::string> &subTreenames = {})
      : fDatasetName(datasetName), fFileNameGlobs(fileNames), fStartEntry(startEntry), fEndEntry(endEntry),
        fDefaultColumns(defaultColumns), fSubTreeNames(subTreenames)
   {
   }
};

} // namespace RDF
} // namespace ROOT

#endif // ROOT_RDF_RDATASETSPEC
