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
   RNTupleCompressor() : fZipBuffer(std::unique_ptr<Buffer_t>(new Buffer_t())) {}
   RNTupleCompressor(const RNTupleCompressor &other) = delete;
   RNTupleCompressor &operator =(const RNTupleCompressor &other) = delete;

   /// Returns the size of the compressed data block. The data is written into the zip buffer
   size_t operator() (const void *from, size_t nbytes, int compression) {
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

   const void *GetZipBuffer() { return fZipBuffer->data(); }
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
   RNTupleDecompressor() : fUnzipBuffer(std::unique_ptr<Buffer_t>(new Buffer_t())) {}
   RNTupleDecompressor(const RNTupleDecompressor &other) = delete;
   RNTupleDecompressor &operator =(const RNTupleDecompressor &other) = delete;

   /**
    * The nbytes parameter provides the size of the from buffer. The dataLen gives the size of the uncompressed data.
    * The block is uncompressed iff nbytes == dataLen.
    */
   void operator() (const void *from, size_t nbytes, size_t dataLen, void *to) {
      if (dataLen == nbytes) {
         memcpy(to, from, nbytes);
         return;
      }
      R__ASSERT(dataLen > nbytes);

      int szUnzipBuffer = dataLen;
      int szSource = nbytes;
      int unzipBytes = 0;
      unsigned char *source = const_cast<unsigned char *>(static_cast<const unsigned char *>(from));
      R__unzip(&szSource, source, &szUnzipBuffer, static_cast<unsigned char *>(to), &unzipBytes);
      R__ASSERT((unzipBytes >= 0) && (static_cast<unsigned int>(unzipBytes) == dataLen));
   }

   /**
    * In-place decompression via unzip buffer
    */
   void operator() (void *fromto, size_t nbytes, size_t dataLen) {
      R__ASSERT(dataLen <= kMAXZIPBUF);
      operator()(fromto, nbytes, dataLen, fUnzipBuffer->data());
      memcpy(fromto, fUnzipBuffer->data(), dataLen);
   }
};

} // namespace Detail
} // namespace Experimental
} // namespace ROOT

#endif
