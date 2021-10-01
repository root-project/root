/// \file ROOT/RNTupleZip.hxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2019-11-21
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RNTupleZip
#define ROOT7_RNTupleZip

#include <RZip.h>
#include <TError.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <functional>
#include <memory>
#include <utility>

namespace ROOT {
namespace Experimental {
namespace Detail {

// clang-format off
/**
\class ROOT::Experimental::Detail::RNTupleCompressor
\ingroup NTuple
\brief Helper class to compress data blocks in the ROOT compression frame format
*/
// clang-format on
class RNTupleCompressor {
private:
   using Buffer_t = std::array<unsigned char, kMAXZIPBUF>;
   std::unique_ptr<Buffer_t> fZipBuffer;

public:
   /// Data might be overwritten, if a zipped block in the middle of a large input data stream
   /// turns out to be uncompressible
   using Writer_t = std::function<void(const void *buffer, size_t nbytes, size_t offset)>;
   static constexpr size_t kMaxSingleBlock = kMAXZIPBUF;

   RNTupleCompressor() : fZipBuffer(std::make_unique<Buffer_t>()) {}
   RNTupleCompressor(const RNTupleCompressor &) = delete;
   RNTupleCompressor &operator =(const RNTupleCompressor &) = delete;
   RNTupleCompressor(RNTupleCompressor &&) = default;
   RNTupleCompressor &operator =(RNTupleCompressor &&) = default;
   ~RNTupleCompressor() = default;

   /// Returns the size of the compressed data. Data is compressed in 16MB (kMAXZIPBUF) blocks and written
   /// piecewise using the provided writer
   size_t Zip(const void *from, size_t nbytes, int compression, Writer_t fnWriter) {
      R__ASSERT(from != nullptr);

      auto cxLevel = compression % 100;
      if (cxLevel == 0) {
         fnWriter(from, nbytes, 0);
         return nbytes;
      }

      auto cxAlgorithm = static_cast<ROOT::RCompressionSetting::EAlgorithm::EValues>(compression / 100);
      unsigned int nZipBlocks = 1 + (nbytes - 1) / kMAXZIPBUF;
      char *source = const_cast<char *>(static_cast<const char *>(from));
      int szTarget = kMAXZIPBUF;
      char *target = reinterpret_cast<char *>(fZipBuffer->data());
      int szOutBlock = 0;
      int szRemaining = nbytes;
      size_t szZipData = 0;
      for (unsigned int i = 0; i < nZipBlocks; ++i) {
         int szSource = std::min(static_cast<int>(kMAXZIPBUF), szRemaining);
         R__zipMultipleAlgorithm(cxLevel, &szSource, source, &szTarget, target, &szOutBlock, cxAlgorithm);
         R__ASSERT(szOutBlock >= 0);
         if ((szOutBlock == 0) || (szOutBlock >= szSource)) {
            // Uncompressible block, we have to store the entire input data stream uncompressed
            fnWriter(from, nbytes, 0);
            return nbytes;
         }

         fnWriter(target, szOutBlock, szZipData);
         szZipData += szOutBlock;
         source += szSource;
         szRemaining -= szSource;
      }
      R__ASSERT(szRemaining == 0);
      R__ASSERT(szZipData < nbytes);
      return szZipData;
   }

   /// Returns the size of the compressed data block. The data is written into the zip buffer.
   /// This works only for small input buffer up to 16MB (kMAXZIPBUF)
   size_t Zip(const void *from, size_t nbytes, int compression) {
      R__ASSERT(from != nullptr);
      R__ASSERT(nbytes <= kMAXZIPBUF);

      auto cxLevel = compression % 100;
      if (cxLevel == 0) {
         memcpy(fZipBuffer->data(), from, nbytes);
         return nbytes;
      }

      auto cxAlgorithm = static_cast<ROOT::RCompressionSetting::EAlgorithm::EValues>(compression / 100);
      int szSource = nbytes;
      char *source = const_cast<char *>(static_cast<const char *>(from));
      int szTarget = nbytes;
      char *target = reinterpret_cast<char *>(fZipBuffer->data());
      int szOut = 0;
      R__zipMultipleAlgorithm(cxLevel, &szSource, source, &szTarget, target, &szOut, cxAlgorithm);
      R__ASSERT(szOut >= 0);
      if ((szOut > 0) && (static_cast<unsigned int>(szOut) < nbytes))
         return szOut;

      memcpy(fZipBuffer->data(), from, nbytes);
      return nbytes;
   }

   /// Returns the size of the compressed data, written into the provided output buffer.
   static std::size_t Zip(const void *from, std::size_t nbytes, int compression, void *to) {
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
         szRemaining -= szSource;
      }
      R__ASSERT(szRemaining == 0);
      R__ASSERT(szZipData < nbytes);
      return szZipData;
   }

   void *GetZipBuffer() { return fZipBuffer->data(); }
};


// clang-format off
/**
\class ROOT::Experimental::Detail::RNTupleDecompressor
\ingroup NTuple
\brief Helper class to uncompress data blocks in the ROOT compression frame format
*/
// clang-format on
class RNTupleDecompressor {
private:
   using Buffer_t = std::array<unsigned char, kMAXZIPBUF>;
   std::unique_ptr<Buffer_t> fUnzipBuffer;

public:
   RNTupleDecompressor() : fUnzipBuffer(std::make_unique<Buffer_t>()) {}
   RNTupleDecompressor(const RNTupleDecompressor &) = delete;
   RNTupleDecompressor &operator =(const RNTupleDecompressor &) = delete;
   RNTupleDecompressor(RNTupleDecompressor &&) = default;
   RNTupleDecompressor &operator =(RNTupleDecompressor &&) = default;
   ~RNTupleDecompressor() = default;

   /**
    * The nbytes parameter provides the size ls of the from buffer. The dataLen gives the size of the uncompressed data.
    * The block is uncompressed iff nbytes == dataLen.
    */
   void Unzip(const void *from, size_t nbytes, size_t dataLen, void *to) {
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

   /**
    * In-place decompression via unzip buffer
    */
   void Unzip(void *fromto, size_t nbytes, size_t dataLen) {
      R__ASSERT(dataLen <= kMAXZIPBUF);
      Unzip(fromto, nbytes, dataLen, fUnzipBuffer->data());
      memcpy(fromto, fUnzipBuffer->data(), dataLen);
   }
};

} // namespace Detail
} // namespace Experimental
} // namespace ROOT

#endif
