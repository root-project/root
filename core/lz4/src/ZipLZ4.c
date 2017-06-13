// Original Author: Brian Bockelman

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ZipLZ4.h"
#include "lz4.h"
#include "lz4hc.h"
#include <stdio.h>
#include <stdint.h>

#include "RConfig.h"

// TODO: Copied from TBranch.cxx
#if (__GNUC__ >= 3) || defined(__INTEL_COMPILER)
#if !defined(R__unlikely)
#define R__unlikely(expr) __builtin_expect(!!(expr), 0)
#endif
#if !defined(R__likely)
#define R__likely(expr) __builtin_expect(!!(expr), 1)
#endif
#else
#define R__unlikely(expr) expr
#define R__likely(expr) expr
#endif

static const int kHeaderSize = 9;

void R__zipLZ4(int cxlevel, int *srcsize, char *src, int *tgtsize, char *tgt, int *irep)
{
   int returnStatus, LZ4_version = LZ4_versionNumber();
   uint64_t out_size; /* compressed size */
   uint64_t in_size = (unsigned)(*srcsize);

   *irep = 0;

   if (*tgtsize <= 0) {
      return;
   }

   if (*srcsize > 0xffffff || *srcsize < 0) {
      return;
   }

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

   tgt[0] = 'L';
   tgt[1] = '4';
   tgt[2] = (LZ4_version / (100 * 100));

   out_size = returnStatus; /* compressed size */

   tgt[3] = (char)(out_size & 0xff);
   tgt[4] = (char)((out_size >> 8) & 0xff);
   tgt[5] = (char)((out_size >> 16) & 0xff);

   tgt[6] = (char)(in_size & 0xff); /* decompressed size */
   tgt[7] = (char)((in_size >> 8) & 0xff);
   tgt[8] = (char)((in_size >> 16) & 0xff);

   *irep = (int)returnStatus + kHeaderSize;
}

void R__unzipLZ4(int *srcsize, unsigned char *src, int *tgtsize, unsigned char *tgt, int *irep)
{
   int returnStatus, LZ4_version = LZ4_versionNumber() / (100 * 100);
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

   returnStatus = LZ4_decompress_safe((char *)(&src[kHeaderSize]), (char *)(tgt), *srcsize - kHeaderSize, *tgtsize);
   if (R__unlikely(returnStatus < 0)) {
      fprintf(stderr, "R__unzipLZ4: error in decompression around byte %d out of maximum %d.\n", -returnStatus,
              *tgtsize);
      return;
   }

   *irep = returnStatus;
}
