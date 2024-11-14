/// \file ROOT/RNTupleMerger.hxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>, Max Orok <maxwellorok@gmail.com>, Alaettin Serhan Mete <amete@anl.gov>
/// \date 2020-07-08
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RNTupleMerger
#define ROOT7_RNTupleMerger

#include <ROOT/RError.hxx>
#include <ROOT/RNTupleDescriptor.hxx>
#include <ROOT/RNTupleUtil.hxx>
#include <ROOT/RPageStorage.hxx>
#include <ROOT/TTaskGroup.hxx>
#include <Compression.h>

#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <unordered_map>

namespace ROOT::Experimental::Internal {

enum class ENTupleMergingMode {
   /// The merger will discard all columns that aren't present in the prototype model (i.e. the model of the first
   /// source)
   kFilter,
   /// The merger will refuse to merge any 2 RNTuples whose schema doesn't match exactly
   kStrict,
   /// The merger will update the output model to include all columns from all sources. Entries corresponding to columns
   /// that are not present in a source will be set to the default value of the type.
   kUnion
};

enum class ENTupleMergeErrBehavior {
   /// The merger will abort merging as soon as an error is encountered
   kAbort,
   /// Upon errors, the merger will skip the current source and continue
   kSkip
};

struct RColumnMergeInfo;
struct RNTupleMergeData;
struct RSealedPageMergeData;

class RClusterPool;

struct RNTupleMergeOptions {
   /// If `fCompressionSettings == kUnknownCompressionSettings` (the default), the merger will not change the
   /// compression of any of its sources (fast merging). Otherwise, all sources will be converted to the specified
   /// compression algorithm and level.
   int fCompressionSettings = kUnknownCompressionSettings;
   /// Determines how the merging treats sources with different models (\see ENTupleMergingMode).
   ENTupleMergingMode fMergingMode = ENTupleMergingMode::kFilter;
   /// Determines how the Merge function behaves upon merging errors
   ENTupleMergeErrBehavior fErrBehavior = ENTupleMergeErrBehavior::kAbort;
   /// If true, the merger will emit further diagnostics and information.
   bool fExtraVerbose = false;
};

// clang-format off
/**
 * \class ROOT::Experimental::Internal::RNTupleMerger
 * \ingroup NTuple
 * \brief Given a set of RPageSources merge them into an RPageSink, optionally changing their compression.
 *        This can also be used to change the compression of a single RNTuple by just passing a single source.
 */
// clang-format on
class RNTupleMerger final {
   std::unique_ptr<RPageAllocator> fPageAlloc;
   std::optional<TTaskGroup> fTaskGroup;

   void MergeCommonColumns(RClusterPool &clusterPool, DescriptorId_t clusterId,
                           std::span<RColumnMergeInfo> commonColumns, const RCluster::ColumnSet_t &commonColumnSet,
                           RSealedPageMergeData &sealedPageData, const RNTupleMergeData &mergeData);

   void MergeSourceClusters(RPageSource &source, std::span<RColumnMergeInfo> commonColumns,
                            std::span<RColumnMergeInfo> extraDstColumns, RNTupleMergeData &mergeData);

public:
   RNTupleMerger();

   /// Merge a given set of sources into the destination.
   RResult<void> Merge(std::span<RPageSource *> sources, RPageSink &destination,
                       const RNTupleMergeOptions &mergeOpts = RNTupleMergeOptions());

}; // end of class RNTupleMerger

} // namespace ROOT::Experimental::Internal

#endif
