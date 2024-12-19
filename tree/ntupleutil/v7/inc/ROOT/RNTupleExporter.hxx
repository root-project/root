/// \file ROOT/RNTupleExporter.hxx
/// \ingroup NTuple ROOT7
/// \author Giacomo Parolini <giacomo.parolini@cern.ch>
/// \date 2024-12-10
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RNTupleExporter
#define ROOT7_RNTupleExporter

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_set>

#include <ROOT/RNTupleUtil.hxx>

namespace ROOT::Experimental::Internal {

class RPageSource;

class RNTupleExporter {
public:
   enum class EFilterType {
      /// Don't export items contained in the filter's set
      kBlacklist,
      /// Export only items contained in the filter's set
      kWhitelist   
   };

   template <typename T>
   struct RFilter {
      std::unordered_set<T> fSet;
      EFilterType fType = EFilterType::kBlacklist;
   };
   
   struct RPagesOptions {
      enum RExportPageFlags {
         kNone = 0x0,
         kIncludeChecksums = 0x1,
         /// If enabled, the exporter will report the current progress on the stderr
         kShowProgressBar = 0x2,

         kDefaults = kShowProgressBar
      };

      std::string fOutputPath;
      std::uint64_t fFlags;

      /// Optional filter that determines which columns are included or excluded from being exported.
      /// By default, export all columns. If you only want to include certain column types, add them
      /// to `fColumnTypeFilter.fSet` and change `fColumnTypeFilter.fType` to kWhitelist.
      RFilter<EColumnType> fColumnTypeFilter;

      RPagesOptions() : fOutputPath("."), fFlags(kDefaults) {}
   };

   struct RPagesResult {
      std::vector<std::string> fExportedFileNames;
   };

   /// Given a page source, writes all its pages to individual files (1 per page).
   /// If the source is not already attached, it will be attached by this process.
   static RPagesResult ExportPages(RPageSource &source, const RPagesOptions &options = {});
};

} // namespace ROOT::Experimental::Internal

#endif
