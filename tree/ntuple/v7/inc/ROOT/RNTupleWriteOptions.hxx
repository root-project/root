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

// clang-format off
/**
\class ROOT::Experimental::RNTupleWriteOptions
\ingroup NTuple
\brief Common user-tunable settings for storing ntuples

All page sink classes need to support the common options.
*/
// clang-format on
class RNTupleWriteOptions {
protected:
   int fCompression{RCompressionSetting::EDefaults::kUseGeneralPurpose};
   /// Approximation of the target compressed cluster size
   std::size_t fApproxZippedClusterSize = 50 * 1000 * 1000;
   /// Memory limit for committing a cluster: with very high compression ratio, we need a limit
   /// on how large the I/O buffer can grow during writing.
   std::size_t fMaxUnzippedClusterSize = 512 * 1024 * 1024;
   /// Should be just large enough so that the compression ratio does not benefit much more from larger pages.
   /// Unless the cluster is too small to contain a sufficiently large page, pages are
   /// fApproxUnzippedPageSize in size and tail pages (the last page in a cluster) is between
   /// fApproxUnzippedPageSize/2 and fApproxUnzippedPageSize * 1.5 in size.
   std::size_t fApproxUnzippedPageSize = 64 * 1024;
   bool fUseBufferedWrite = true;
   /// If set, 64bit index columns are replaced by 32bit index columns. This limits the cluster size to 512MB
   /// but it can result in smaller file sizes for data sets with many collections and lz4 or no compression.
   bool fHasSmallClusters = false;

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

   bool GetUseBufferedWrite() const { return fUseBufferedWrite; }
   void SetUseBufferedWrite(bool val) { fUseBufferedWrite = val; }

   bool GetHasSmallClusters() const { return fHasSmallClusters; }
   void SetHasSmallClusters(bool val) { fHasSmallClusters = val; }
};

} // namespace Experimental
} // namespace ROOT

#endif // ROOT7_RNTupleWriteOptions
