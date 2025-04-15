/// \file ROOT/RNTupleZip.hxx
/// \ingroup NTuple
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

   /// Returns the size of the compressed data, written into the provided output buffer.
   static std::size_t Zip(const void *from, std::size_t nbytes, int compression, void *to)
   {
      R__ASSERT(from != nullptr);
      R__ASSERT(to != nullptr);
      auto cxLevel = compression % 100;
      if (cxLevel == 0) {
         memcpy(to, from, nbytes);
         return nbytes;
      }

      auto cxAlgorithm = static_cast<ROOT::RCompressionSetting::EAlgorithm::EValues>(compression / 100);
      unsigned int nZipBlocks = 1 + (nbytes - 1) / kMAXZIPBUF;
      char *source = const_cast<char *>(static_cast<const char *>(from));
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
            return nbytes;
         }

         szZipData += szOutBlock;
         source += szSource;
         target += szOutBlock;
         szRemaining -= szSource;
      }
      R__ASSERT(szRemaining == 0);
      R__ASSERT(szZipData < nbytes);
      return szZipData;
   }
};

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

      unsigned char *source = const_cast<unsigned char *>(static_cast<const unsigned char *>(from));
      unsigned char *target = static_cast<unsigned char *>(to);
      int szRemaining = dataLen;
      do {
         int szSource;
         int szTarget;
         int retval = R__unzip_header(&szSource, source, &szTarget);
         R__ASSERT(retval == 0);
         R__ASSERT(szSource > 0);
         R__ASSERT(szTarget > szSource);
         R__ASSERT(static_cast<unsigned int>(szSource) <= nbytes);
         R__ASSERT(static_cast<unsigned int>(szTarget) <= dataLen);

         int unzipBytes = 0;
         R__unzip(&szSource, source, &szTarget, target, &unzipBytes);
         R__ASSERT(unzipBytes == szTarget);

         target += szTarget;
         source += szSource;
         szRemaining -= unzipBytes;
      } while (szRemaining > 0);
      R__ASSERT(szRemaining == 0);
   }
};

} // namespace Internal
} // namespace ROOT

#endif
