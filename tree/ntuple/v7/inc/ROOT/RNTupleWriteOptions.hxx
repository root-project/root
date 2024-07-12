/// \file ROOT/RNTupleWriteOptions.hxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2024-02-22
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RNTupleWriteOptions
#define ROOT7_RNTupleWriteOptions

#include <Compression.h>

#include <cstdint>
#include <cstddef>
#include <memory>

namespace ROOT {
namespace Experimental {

class RNTupleWriteOptions;

namespace Internal {

class RNTupleWriteOptionsManip final {
public:
   static void SetMaxKeySize(RNTupleWriteOptions &options, std::uint64_t maxKeySize);
};

} // namespace Internal

// clang-format off
/**
\class ROOT::Experimental::RNTupleWriteOptions
\ingroup NTuple
\brief Common user-tunable settings for storing ntuples

All page sink classes need to support the common options.
*/
// clang-format on
class RNTupleWriteOptions {
public:
   enum class EImplicitMT {
      kOff,
      kDefault,
   };

   // clang-format off
   static constexpr std::uint64_t kDefaultMaxKeySize = 0x4000'0000; // 1 GiB

   friend Internal::RNTupleWriteOptionsManip;
   // clang-format on

protected:
   int fCompression{RCompressionSetting::EDefaults::kUseGeneralPurpose};
   /// Approximation of the target compressed cluster size
   std::size_t fApproxZippedClusterSize = 50 * 1000 * 1000;
   /// Memory limit for committing a cluster: with very high compression ratio, we need a limit
   /// on how large the I/O buffer can grow during writing.
   std::size_t fMaxUnzippedClusterSize = 512 * 1024 * 1024;
   /// Should be just large enough so that the compression ratio does not benefit much more from larger pages.
   /// Unless the cluster is too small to contain a sufficiently large page, pages are
   /// fApproxUnzippedPageSize in size. If tail page optimization is enabled, the last page in a cluster is
   /// between fApproxUnzippedPageSize/2 and fApproxUnzippedPageSize * 1.5 in size.
   std::size_t fApproxUnzippedPageSize = 64 * 1024;
   /// Whether to optimize tail pages to avoid an undersized last page per cluster (see above). Increases the
   /// required memory by a factor 3x.
   bool fUseTailPageOptimization = true;
   /// Whether to use buffered writing (with RPageSinkBuf). This buffers compressed pages in memory, reorders them
   /// to keep pages of the same column adjacent, and coalesces the writes when committing a cluster.
   bool fUseBufferedWrite = true;
   /// Whether to use implicit multi-threading to compress pages. Only has an effect if buffered writing is turned on.
   EImplicitMT fUseImplicitMT = EImplicitMT::kDefault;
   /// If set, 64bit index columns are replaced by 32bit index columns. This limits the cluster size to 512MB
   /// but it can result in smaller file sizes for data sets with many collections and lz4 or no compression.
   bool fHasSmallClusters = false;
   /// If set, checksums will be calculated and written for every page.
   bool fEnablePageChecksums = true;
   /// Specifies the max size of a payload storeable into a single TKey. When writing an RNTuple to a ROOT file,
   /// any payload whose size exceeds this will be split into multiple keys.
   std::uint64_t fMaxKeySize = kDefaultMaxKeySize;

public:
   /// A maximum size of 512MB still allows for a vector of bool to be stored in a small cluster.  This is the
   /// worst case wrt. the maximum required size of the index column.  A 32bit index column can address 512MB
   /// of 1-bit (on disk size) bools.
   static constexpr std::uint64_t kMaxSmallClusterSize = 512 * 1024 * 1024;

   virtual ~RNTupleWriteOptions() = default;
   virtual std::unique_ptr<RNTupleWriteOptions> Clone() const;

   int GetCompression() const { return fCompression; }
   void SetCompression(int val) { fCompression = val; }
   void SetCompression(RCompressionSetting::EAlgorithm::EValues algorithm, int compressionLevel)
   {
      fCompression = CompressionSettings(algorithm, compressionLevel);
   }

   std::size_t GetApproxZippedClusterSize() const { return fApproxZippedClusterSize; }
   void SetApproxZippedClusterSize(std::size_t val);

   std::size_t GetMaxUnzippedClusterSize() const { return fMaxUnzippedClusterSize; }
   void SetMaxUnzippedClusterSize(std::size_t val);

   std::size_t GetApproxUnzippedPageSize() const { return fApproxUnzippedPageSize; }
   void SetApproxUnzippedPageSize(std::size_t val);

   bool GetUseTailPageOptimization() const { return fUseTailPageOptimization; }
   void SetUseTailPageOptimization(bool val) { fUseTailPageOptimization = val; }

   bool GetUseBufferedWrite() const { return fUseBufferedWrite; }
   void SetUseBufferedWrite(bool val) { fUseBufferedWrite = val; }

   EImplicitMT GetUseImplicitMT() const { return fUseImplicitMT; }
   void SetUseImplicitMT(EImplicitMT val) { fUseImplicitMT = val; }

   bool GetHasSmallClusters() const { return fHasSmallClusters; }
   void SetHasSmallClusters(bool val) { fHasSmallClusters = val; }

   bool GetEnablePageChecksums() const { return fEnablePageChecksums; }
   /// Note that turning off page checksums will also turn off the same page merging optimization (see tuning.md)
   void SetEnablePageChecksums(bool val) { fEnablePageChecksums = val; }

   std::uint64_t GetMaxKeySize() const { return fMaxKeySize; }
};

namespace Internal {
inline void RNTupleWriteOptionsManip::SetMaxKeySize(RNTupleWriteOptions &options, std::uint64_t maxKeySize)
{
   options.fMaxKeySize = maxKeySize;
}
} // namespace Internal

} // namespace Experimental
} // namespace ROOT

#endif // ROOT7_RNTupleWriteOptions
