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

class RNTupleWriteOptions;

namespace Internal {

class RNTupleWriteOptionsManip final {
public:
   static void SetMaxKeySize(RNTupleWriteOptions &options, std::uint64_t maxKeySize);
};

} // namespace Internal

// clang-format off
/**
\class ROOT::RNTupleWriteOptions
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
   std::uint32_t fCompression{RCompressionSetting::EDefaults::kUseGeneralPurpose};
   /// Approximation of the target compressed cluster size
   std::size_t fApproxZippedClusterSize = 128 * 1024 * 1024;
   /// Memory limit for committing a cluster: with very high compression ratio, we need a limit
   /// on how large the I/O buffer can grow during writing.
   std::size_t fMaxUnzippedClusterSize = 10 * fApproxZippedClusterSize;
   /// Initially, columns start with a page of this size. The default value is chosen to accomodate at least 32 elements
   /// of 64 bits, or 64 elements of 32 bits. If more elements are needed, pages are increased up until the byte limit
   /// given by fMaxUnzippedPageSize or until the total page buffer limit is reached (as a sum of all page buffers).
   /// The total write buffer limit needs to be large enough to hold the initial pages of all columns.
   std::size_t fInitialUnzippedPageSize = 256;
   /// Pages can grow only to the given limit in bytes.
   std::size_t fMaxUnzippedPageSize = 1024 * 1024;
   /// The maximum size that the sum of all page buffers used for writing into a persistent sink are allowed to use.
   /// If set to zero, RNTuple will auto-adjust the budget based on the value of fApproxZippedClusterSize.
   /// If set manually, the size needs to be large enough to hold all initial page buffers.
   /// The total amount of memory for writing is larger, e.g. for the additional compressed buffers etc.
   /// Use RNTupleModel::EstimateWriteMemoryUsage() for the total estimated memory use for writing.
   /// The default values are tuned for a total write memory of around 300 MB per fill context.
   std::size_t fPageBufferBudget = 0;
   /// Whether to use buffered writing (with RPageSinkBuf). This buffers compressed pages in memory, reorders them
   /// to keep pages of the same column adjacent, and coalesces the writes when committing a cluster.
   bool fUseBufferedWrite = true;
   /// Whether to use Direct I/O for writing. Note that this introduces alignment requirements that may very between
   /// filesystems and platforms.
   bool fUseDirectIO = false;
   /// Buffer size to use for writing to files, must be a multiple of 4096 bytes. Testing suggests that 4MiB gives best
   /// performance (with Direct I/O) at a reasonable memory consumption.
   std::size_t fWriteBufferSize = 4 * 1024 * 1024;
   /// Whether to use implicit multi-threading to compress pages. Only has an effect if buffered writing is turned on.
   EImplicitMT fUseImplicitMT = EImplicitMT::kDefault;
   /// If set, checksums will be calculated and written for every page.
   bool fEnablePageChecksums = true;
   /// If set, identical pages are deduplicated and aliased on disk. Requires page checksums.
   bool fEnableSamePageMerging = true;
   /// Specifies the max size of a payload storeable into a single TKey. When writing an RNTuple to a ROOT file,
   /// any payload whose size exceeds this will be split into multiple keys.
   std::uint64_t fMaxKeySize = kDefaultMaxKeySize;

public:

   virtual ~RNTupleWriteOptions() = default;
   virtual std::unique_ptr<RNTupleWriteOptions> Clone() const;

   std::uint32_t GetCompression() const { return fCompression; }
   void SetCompression(std::uint32_t val) { fCompression = val; }
   void SetCompression(RCompressionSetting::EAlgorithm::EValues algorithm, int compressionLevel)
   {
      fCompression = CompressionSettings(algorithm, compressionLevel);
   }

   std::size_t GetApproxZippedClusterSize() const { return fApproxZippedClusterSize; }
   void SetApproxZippedClusterSize(std::size_t val);

   std::size_t GetMaxUnzippedClusterSize() const { return fMaxUnzippedClusterSize; }
   void SetMaxUnzippedClusterSize(std::size_t val);

   std::size_t GetInitialUnzippedPageSize() const { return fInitialUnzippedPageSize; }
   void SetInitialUnzippedPageSize(std::size_t val);

   std::size_t GetMaxUnzippedPageSize() const { return fMaxUnzippedPageSize; }
   void SetMaxUnzippedPageSize(std::size_t val);

   std::size_t GetPageBufferBudget() const;
   void SetPageBufferBudget(std::size_t val) { fPageBufferBudget = val; }

   bool GetUseBufferedWrite() const { return fUseBufferedWrite; }
   void SetUseBufferedWrite(bool val) { fUseBufferedWrite = val; }

   bool GetUseDirectIO() const { return fUseDirectIO; }
   void SetUseDirectIO(bool val) { fUseDirectIO = val; }

   std::size_t GetWriteBufferSize() const { return fWriteBufferSize; }
   void SetWriteBufferSize(std::size_t val) { fWriteBufferSize = val; }

   EImplicitMT GetUseImplicitMT() const { return fUseImplicitMT; }
   void SetUseImplicitMT(EImplicitMT val) { fUseImplicitMT = val; }

   bool GetEnablePageChecksums() const { return fEnablePageChecksums; }
   /// Note that turning off page checksums will also turn off the same page merging optimization (see tuning.md)
   void SetEnablePageChecksums(bool val)
   {
      fEnablePageChecksums = val;
      if (!fEnablePageChecksums) {
         fEnableSamePageMerging = false;
      }
   }

   bool GetEnableSamePageMerging() const { return fEnableSamePageMerging; }
   void SetEnableSamePageMerging(bool val);

   std::uint64_t GetMaxKeySize() const { return fMaxKeySize; }
};

namespace Internal {
inline void RNTupleWriteOptionsManip::SetMaxKeySize(RNTupleWriteOptions &options, std::uint64_t maxKeySize)
{
   options.fMaxKeySize = maxKeySize;
}

} // namespace Internal

namespace Experimental {
// TODO(gparolini): remove before branching ROOT v6.36
using RNTupleWriteOptions [[deprecated("ROOT::Experimental::RNTupleWriteOptions moved to ROOT::RNTupleWriteOptions")]] =
   ROOT::RNTupleWriteOptions;
} // namespace Experimental

} // namespace ROOT

#endif // ROOT7_RNTupleWriteOptions
