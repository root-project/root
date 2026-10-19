/// \file ROOT/RNTupleZip.hxx
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2019-11-21

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RNTupleZip
#define ROOT_RNTupleZip

#include <RZip.h>
#include <TError.h>

#include <ROOT/RError.hxx>
#include <RConfigure.h>

#ifdef R__HAS_LHC4CODEC
#include <ZipLHC4.h>
#endif

#include <algorithm>
#include <array>
#include <cstring>
#include <functional>
#include <memory>
#include <utility>

namespace ROOT {
namespace Internal {

// clang-format off
/**
\class ROOT::Internal::RNTupleDecompressor
\ingroup NTuple
\brief Helper class to uncompress data blocks in the ROOT compression frame format
*/
// clang-format on
class RNTupleDecompressor {
public:
   RNTupleDecompressor() = delete;
   RNTupleDecompressor(const RNTupleDecompressor &other) = delete;
   RNTupleDecompressor &operator=(const RNTupleDecompressor &other) = delete;
   RNTupleDecompressor(RNTupleDecompressor &&other) = delete;
   RNTupleDecompressor &operator=(RNTupleDecompressor &&other) = delete;

   /**
    * The nbytes parameter provides the size ls of the from buffer. The dataLen gives the size of the uncompressed data.
    * The block is uncompressed iff nbytes == dataLen.
    */
   static void Unzip(const void *from, size_t nbytes, size_t dataLen, void *to)
   {
      if (dataLen == nbytes) {
         memcpy(to, from, nbytes);
         return;
      }
      R__ASSERT(dataLen > nbytes);

      const unsigned char *source = static_cast<const unsigned char *>(from);
      unsigned char *target = static_cast<unsigned char *>(to);
      size_t szRemainingLen = dataLen;
      size_t szRemainingNbytes = nbytes;
      do {
         if (R__unlikely(szRemainingNbytes < ROOT::Internal::kZipHeaderSize)) {
            throw ROOT::RException(R__FAIL("zip buffer too short"));
         }
         int szSource;
         int szTarget;
         int retval = R__unzip_header(&szSource, source, &szTarget);
         if (R__unlikely(!((retval == 0) && (szSource > 0) && (szTarget > szSource) &&
                           (static_cast<unsigned int>(szSource) <= szRemainingNbytes) &&
                           (static_cast<unsigned int>(szTarget) <= szRemainingLen)))) {
            throw ROOT::RException(R__FAIL("failed to unzip buffer header"));
         }

         int unzipBytes = 0;
         R__unzip(&szSource, source, &szTarget, target, &unzipBytes);
         if (R__unlikely(unzipBytes != szTarget)) {
            throw ROOT::RException(R__FAIL(std::string("unexpected length after unzipping the buffer (wanted: ") +
                                           std::to_string(szTarget) + ", got: " + std::to_string(unzipBytes) + ")"));
         }

         target += szTarget;
         source += szSource;
         szRemainingNbytes -= szSource;
         szRemainingLen -= unzipBytes;
      } while (szRemainingLen > 0);
      R__ASSERT(szRemainingLen == 0);
      if (szRemainingNbytes > 0) {
         throw ROOT::RException(R__FAIL(std::string("unexpected trailing bytes in zip buffer")));
      }
   }
};

inline void VerifyZipRoundtrip(const void *from, std::size_t nbytes, const void *to, std::size_t szZipData)
{
   if (nbytes == szZipData) {
      if (std::memcmp(from, to, nbytes) != 0)
         throw ROOT::RException(R__FAIL("zip verify: uncompressed page buffer mismatch"));
      return;
   }

#ifdef R__HAS_LHC4CODEC
   auto dumpLhc4VerifyFailure = [&](const char *stage, const char *status) {
      if (szZipData < 2)
         return;
      const auto *hdr = static_cast<const unsigned char *>(to);
      if (hdr[0] != 'L' || hdr[1] != 'C')
         return;
      R__LHC4FailureDumpInfo dump{};
      dump.fStage = stage;
      dump.fUncompressed = from;
      dump.fUncompressedSize = nbytes;
      dump.fCompressedLc = to;
      dump.fCompressedLcSize = szZipData;
      dump.fCxLevel = -1;
      dump.fStatusMessage = status;
      R__DumpLHC4Failure(&dump);
   };
#endif

   const auto roundtrip = std::make_unique<char[]>(nbytes);
   try {
      RNTupleDecompressor::Unzip(to, szZipData, nbytes, roundtrip.get());
   } catch (const ROOT::RException &err) {
#ifdef R__HAS_LHC4CODEC
      dumpLhc4VerifyFailure("verify_unzip", err.what());
#endif
      throw;
   }
   if (std::memcmp(from, roundtrip.get(), nbytes) != 0) {
#ifdef R__HAS_LHC4CODEC
      dumpLhc4VerifyFailure("verify_mismatch", "decompressed page buffer mismatch");
#endif
      throw ROOT::RException(R__FAIL("zip verify: decompressed page buffer mismatch"));
   }
}

// clang-format off
/**
\class ROOT::Internal::RNTupleCompressor
\ingroup NTuple
\brief Helper class to compress data blocks in the ROOT compression frame format
*/
// clang-format on
class RNTupleCompressor {
public:
   RNTupleCompressor() = delete;
   RNTupleCompressor(const RNTupleCompressor &other) = delete;
   RNTupleCompressor &operator=(const RNTupleCompressor &other) = delete;
   RNTupleCompressor(RNTupleCompressor &&other) = delete;
   RNTupleCompressor &operator=(RNTupleCompressor &&other) = delete;

   /// When enabled, every Zip() round-trips through Unzip() and compares page bytes.
   static void SetVerifyRoundtrip(bool enable) { fVerifyRoundtrip = enable; }
   static bool GetVerifyRoundtrip() { return fVerifyRoundtrip; }

   /// Returns the size of the compressed data, written into the provided output buffer.
   static std::size_t Zip(const void *from, std::size_t nbytes, int compression, void *to)
   {
      R__ASSERT(from != nullptr);
      R__ASSERT(to != nullptr);
      auto cxLevel = compression % 100;
      if (cxLevel == 0) {
         memcpy(to, from, nbytes);
         if (fVerifyRoundtrip)
            VerifyZipRoundtrip(from, nbytes, to, nbytes);
         return nbytes;
      }

      auto cxAlgorithm = static_cast<ROOT::RCompressionSetting::EAlgorithm::EValues>(compression / 100);
      unsigned int nZipBlocks = 1 + (nbytes - 1) / kMAXZIPBUF;
      const char *source = static_cast<const char *>(from);
      int szTarget = nbytes;
      char *target = reinterpret_cast<char *>(to);
      int szOutBlock = 0;
      int szRemaining = nbytes;
      size_t szZipData = 0;
      for (unsigned int i = 0; i < nZipBlocks; ++i) {
         int szSource = std::min(static_cast<int>(kMAXZIPBUF), szRemaining);
         R__zipMultipleAlgorithm(cxLevel, &szSource, source, &szTarget, target, &szOutBlock, cxAlgorithm);
         R__ASSERT(szOutBlock >= 0);
         if ((szOutBlock == 0) || (szOutBlock >= szSource)) {
            // Uncompressible block, we have to store the entire input data stream uncompressed
            memcpy(to, from, nbytes);
            if (fVerifyRoundtrip)
               VerifyZipRoundtrip(from, nbytes, to, nbytes);
            return nbytes;
         }

         szZipData += szOutBlock;
         source += szSource;
         target += szOutBlock;
         szRemaining -= szSource;
      }
      R__ASSERT(szRemaining == 0);
      R__ASSERT(szZipData < nbytes);
      if (fVerifyRoundtrip)
         VerifyZipRoundtrip(from, nbytes, to, szZipData);
      return szZipData;
   }

private:
   inline static bool fVerifyRoundtrip = false;
};

} // namespace Internal
} // namespace ROOT

#endif
