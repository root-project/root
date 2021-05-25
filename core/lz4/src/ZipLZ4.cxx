// Original Author: Brian Bockelman

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ZipLZ4.h"

#include "ROOT/RConfig.hxx"

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <lz4.h>
#include <lz4hc.h>
#include <xxhash.h>

// Header consists of:
// - 2 byte identifier "L4"
// - 1 byte LZ4 version string.
// - 3 bytes of uncompressed size
// - 3 bytes of compressed size
// - 8 byte checksum using xxhash 64.
static const int kChecksumOffset = 2 + 1 + 3 + 3;
static const int kChecksumSize = sizeof(XXH64_canonical_t);
static const int kHeaderSize = kChecksumOffset + kChecksumSize;

void R__zipLZ4(int cxlevel, int *srcsize, char *src, int *tgtsize, char *tgt, int *irep)
{
   int LZ4_version = LZ4_versionNumber();
   uint64_t out_size; /* compressed size */
   uint64_t in_size = (unsigned)(*srcsize);

   *irep = 0;

   if (R__unlikely(*tgtsize <= 0)) {
      return;
   }

   // Refuse to compress more than 16MB at a time -- we are only allowed 3 bytes for size info.
   if (R__unlikely(*srcsize > 0xffffff || *srcsize < 0)) {
      return;
   }

   int returnStatus;
   if (cxlevel > 9) {
      cxlevel = 9;
   }
   if (cxlevel >= 4) {
      returnStatus = LZ4_compress_HC(src, &tgt[kHeaderSize], *srcsize, *tgtsize - kHeaderSize, cxlevel);
   } else {
      returnStatus = LZ4_compress_default(src, &tgt[kHeaderSize], *srcsize, *tgtsize - kHeaderSize);
   }

   if (R__unlikely(returnStatus == 0)) { /* LZ4 compression failed */
      return;
   }
   XXH64_hash_t checksumResult = XXH64(tgt + kHeaderSize, returnStatus, 0);

   tgt[0] = 'L';
   tgt[1] = '4';
   tgt[2] = (LZ4_version / (100 * 100));

   out_size = returnStatus + kChecksumSize; /* compressed size, including the checksum. */

   // NOTE: these next 6 bytes are required from the ROOT compressed buffer format;
   // upper layers will assume they are laid out in a specific manner.
   tgt[3] = (char)(out_size & 0xff);
   tgt[4] = (char)((out_size >> 8) & 0xff);
   tgt[5] = (char)((out_size >> 16) & 0xff);

   tgt[6] = (char)(in_size & 0xff); /* decompressed size */
   tgt[7] = (char)((in_size >> 8) & 0xff);
   tgt[8] = (char)((in_size >> 16) & 0xff);

   // Write out checksum.
   XXH64_canonicalFromHash(reinterpret_cast<XXH64_canonical_t *>(tgt + kChecksumOffset), checksumResult);

   *irep = (int)returnStatus + kHeaderSize;
}

void R__unzipLZ4(int *srcsize, unsigned char *src, int *tgtsize, unsigned char *tgt, int *irep)
{
   // NOTE: We don't check that srcsize / tgtsize is reasonable or within the ROOT-imposed limits.
   // This is assumed to be handled by the upper layers.

   int LZ4_version = LZ4_versionNumber() / (100 * 100);
   *irep = 0;
   if (R__unlikely(src[0] != 'L' || src[1] != '4')) {
      fprintf(stderr, "R__unzipLZ4: algorithm run against buffer with incorrect header (got %d%d; expected %d%d).\n",
              src[0], src[1], 'L', '4');
      return;
   }
   if (R__unlikely(src[2] != LZ4_version)) {
      fprintf(stderr,
              "R__unzipLZ4: This version of LZ4 is incompatible with the on-disk version (got %d; expected %d).\n",
              src[2], LZ4_version);
      return;
   }

   int inputBufferSize = *srcsize - kHeaderSize;

   // TODO: The checksum followed by the decompression means we iterate through the buffer twice.
   // We should perform some performance tests to see whether we can interleave the two -- i.e., at
   // what size of chunks does interleaving (avoiding two fetches from RAM) improve enough for the
   // extra function call costs?  NOTE that ROOT limits the buffer size to 16MB.
   XXH64_hash_t checksumResult = XXH64(src + kHeaderSize, inputBufferSize, 0);
   XXH64_hash_t checksumFromFile =
      XXH64_hashFromCanonical(reinterpret_cast<XXH64_canonical_t *>(src + kChecksumOffset));

   if (R__unlikely(checksumFromFile != checksumResult)) {
      fprintf(
         stderr,
         "R__unzipLZ4: Buffer corruption error!  Calculated checksum %llu; checksum calculated in the file was %llu.\n",
         (unsigned long long) checksumResult, (unsigned long long) checksumFromFile);
      return;
   }
   int returnStatus = LZ4_decompress_safe((char *)(&src[kHeaderSize]), (char *)(tgt), inputBufferSize, *tgtsize);
   if (R__unlikely(returnStatus < 0)) {
      fprintf(stderr, "R__unzipLZ4: error in decompression around byte %d out of maximum %d.\n", -returnStatus,
              *tgtsize);
      return;
   }

   *irep = returnStatus;
}
