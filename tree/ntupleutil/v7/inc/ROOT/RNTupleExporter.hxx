/// \file ROOT/RNTupleExporter.hxx
/// \ingroup NTuple ROOT7
/// \author Giacomo Parolini <giacomo.parolini@cern.ch>
/// \date 2024-12-10
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RNTupleExporter
#define ROOT7_RNTupleExporter

#include <filesystem>
#include <vector>

namespace ROOT::Experimental::Internal {

class RPageSource;

struct RExportPagesOptions {
   enum RExportPageFlags {
      kNone = 0x0,
      kIncludeChecksums = 0x1,   
      kReportExportedFileNames = 0x2,

      kDefaults = kReportExportedFileNames
   };

   std::filesystem::path fOutputPath = ".";
   std::uint64_t fFlags = kDefaults;
};

struct RExportPagesResult {
   int fNPagesExported;
   // This only gets filled if kReportExportedFileNames was part of the options flags
   std::vector<std::string> fExportedFileNames;
};

class RNTupleExporter {
public:
   /// Given a page source, writes all its pages to individual files (1 per page).
   /// If the source is not already attached, it will be attached by this process.
   static RExportPagesResult ExportPages(RPageSource &source, const RExportPagesOptions &options = {});
};

} // namespace ROOT::Experimental::Internal

#endif
