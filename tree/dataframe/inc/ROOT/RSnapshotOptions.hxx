// Author: Guilherme Amadio, Enrico Guiraud, Danilo Piparo CERN  2/2018

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RSNAPSHOTOPTIONS
#define ROOT_RSNAPSHOTOPTIONS

#include "ROOT/RNTupleWriteOptions.hxx"
#include <Compression.h>

#include <string_view>
#include <string>

namespace ROOT {

namespace RDF {
enum class ESnapshotOutputFormat {
   kDefault,
   kTTree,
   kRNTuple
};

/// A collection of options to steer the creation of the dataset on file
///
/// Some settings are output format-dependent. If `fOutputFormat` is set to `ESnapshotOutputFormat::kTTree`, the
/// `fNTupleWriteOpts` option will be ignored. If `fOutputFormat` is set to `ESnapshotOutputFormat::kRNTuple`,
/// `fAutoFlush`, `fSplitLevel` and `fBasketSize` will be ignored.
///
/// When `fOutputFormat` is set to `ESnapshotOutputFormat::kRNTuple`, fCompressionAlgorithm and fCompressionLevel will
/// be used, *unless* the compression settings in `fNTupleWriteOpts` have been set to something other than the default
/// (through RNTupleWriteOptions::SetCompression()).
struct RSnapshotOptions {
   using ECAlgo = ROOT::RCompressionSetting::EAlgorithm::EValues;
   RSnapshotOptions() = default;
   RSnapshotOptions(std::string_view mode, ECAlgo comprAlgo, int comprLevel, int autoFlush, int splitLevel, bool lazy,
                    bool overwriteIfExists = false, bool vector2RVec = true, int basketSize = -1,
                    ROOT::RNTupleWriteOptions ntupleWriteOpts = ROOT::RNTupleWriteOptions(),
                    ESnapshotOutputFormat outputFormat = ESnapshotOutputFormat::kDefault)
      : fMode(mode),
        fCompressionAlgorithm(comprAlgo),
        fCompressionLevel{comprLevel},
        fAutoFlush(autoFlush),
        fSplitLevel(splitLevel),
        fLazy(lazy),
        fOverwriteIfExists(overwriteIfExists),
        fVector2RVec(vector2RVec),
        fBasketSize(basketSize),
        fNTupleWriteOpts(ntupleWriteOpts),
        fOutputFormat(outputFormat)
   {
   }
   std::string fMode = "RECREATE"; ///< Mode of creation of output file
   ECAlgo fCompressionAlgorithm =
      ROOT::RCompressionSetting::EAlgorithm::kZSTD; ///< Compression algorithm of output file
   int fCompressionLevel = 5;                       ///< Compression level of output file
   int fAutoFlush = 0;                              ///< AutoFlush value for output tree
   int fSplitLevel = 99;                            ///< Split level of output tree
   bool fLazy = false;                              ///< Do not start the event loop when Snapshot is called
   bool fOverwriteIfExists = false;  ///< If fMode is "UPDATE", overwrite object in output file if it already exists
   bool fVector2RVec = true;         ///< If set to true will convert std::vector columns to RVec when saving to disk
   bool fIncludeVariations = false;  ///< Include columns that result from a Vary() action
   int fBasketSize = -1;             ///< Set a custom basket size option. For more details, see
                                     ///< https://root.cern/manual/trees/#baskets-clusters-and-the-tree-header
   ROOT::RNTupleWriteOptions fNTupleWriteOpts = ROOT::RNTupleWriteOptions(); ///< RNTuple-specific write options
   ESnapshotOutputFormat fOutputFormat = ESnapshotOutputFormat::kDefault;    ///< Which data format to write to
};
} // namespace RDF
} // namespace ROOT

#endif
