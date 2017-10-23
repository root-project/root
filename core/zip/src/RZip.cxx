/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "Compression.h"
#include "RConfigure.h"
#include "RZip.h"
#include "Bits.h"
#include "ZipLZMA.h"
#include "ZipLZ4.h"

#include "zlib.h"

#include <stdio.h>
#include <assert.h>

/**
 * Forward decl's
 */
static void R__zipOld(int cxlevel, int *srcsize, char *src, int *tgtsize, char *tgrt, int *irep);
static void R__zipZLIB(int cxlevel, int *srcsize, char *src, int *tgtsize, char *tgrt, int *irep);

/* ===========================================================================
   R__ZipMode is used to select the compression algorithm when R__zip is called
   and when R__zipMultipleAlgorithm is called with its last argument set to 0.
   R__ZipMode = 1 : ZLIB compression algorithm is used (default)
   R__ZipMode = 2 : LZMA compression algorithm is used
   R__ZipMode = 4 : LZ4  compression algorithm is used
   R__ZipMode = 0 or 3 : a very old compression algorithm is used
   (the very old algorithm is supported for backward compatibility)
   The LZMA algorithm requires the external XZ package be installed when linking
   is done. LZMA typically has significantly higher compression factors, but takes
   more CPU time and memory resources while compressing.

  The LZ4 algorithm requires the external LZ4 package to be installed when linking
  is done.  LZ4 typically has the worst compression ratios, but much faster decompression
  speeds - sometimes by an order of magnitude.
*/
enum ROOT::ECompressionAlgorithm R__ZipMode = ROOT::ECompressionAlgorithm::kZLIB;

/* ===========================================================================
   Function to set the ZipMode
 */
extern "C" void R__SetZipMode(enum ROOT::ECompressionAlgorithm mode)
{
   R__ZipMode = mode;
}

unsigned long R__crc32(unsigned long crc, const unsigned char* buf, unsigned int len)
{
   return crc32(crc, buf, len);
}

#define HDRSIZE 9

/* int  *srcsize, *tgtsize, *irep;   source and target sizes, replay */
/* char *tgt, *src;                  source and target buffers */
/* compressionAlgorithm 0 = use global setting */
/*                      1 = zlib */
/*                      2 = lzma */
/*                      3 = old */
void R__zipMultipleAlgorithm(int cxlevel, int *srcsize, char *src, int *tgtsize, char *tgt, int *irep, ROOT::ECompressionAlgorithm compressionAlgorithm)
     /* int cxlevel;                      compression level */
{

  if (*srcsize < 1 + HDRSIZE + 1) {
     *irep = 0;
     return;
  }

  if (cxlevel <= 0) {
    *irep = 0;
    return;
  }

  if (compressionAlgorithm == ROOT::ECompressionAlgorithm::kUseGlobalCompressionSetting) {
    compressionAlgorithm = R__ZipMode;
  }

  // The LZMA compression algorithm from the XZ package
  if (compressionAlgorithm == ROOT::ECompressionAlgorithm::kLZMA) {
     R__zipLZMA(cxlevel, srcsize, src, tgtsize, tgt, irep);
     return;
  } else if (compressionAlgorithm == ROOT::ECompressionAlgorithm::kLZ4) {
     R__zipLZ4(cxlevel, srcsize, src, tgtsize, tgt, irep);
     return;
  } else if (compressionAlgorithm == ROOT::ECompressionAlgorithm::kOldCompressionAlgo || compressionAlgorithm == ROOT::ECompressionAlgorithm::kUseGlobalCompressionSetting) {
     R__zipOld(cxlevel, srcsize, src, tgtsize, tgt, irep);
     return;
  } else {
     // 1 is for ZLIB (which is the default), ZLIB is also used for any illegal
     // algorithm setting.  This was a poor historic choice, as poor code may result in
     // a surprising change in algorithm in a future version of ROOT.
     R__zipZLIB(cxlevel, srcsize, src, tgtsize, tgt, irep);
     return;
  }
}

  // The very old algorithm for backward compatibility
  // 0 for selecting with R__ZipMode in a backward compatible way
  // 3 for selecting in other cases
static void R__zipOld(int cxlevel, int *srcsize, char *src, int *tgtsize, char *tgt, int *irep)
{
    int method   = Z_DEFLATED;
    bits_internal_state state;
    ush att      = (ush)UNKNOWN;
    ush flags    = 0;
    if (cxlevel > 9) cxlevel = 9;
    gCompressionLevel = cxlevel;

    *irep        = 0;
    /* error_flag   = 0; */
    if (*tgtsize <= 0) {
       R__error("target buffer too small");
       return;
    }
    if (*srcsize > 0xffffff) {
       R__error("source buffer too big");
       return;
    }

#ifdef DYN_ALLOC
    state.R__window = 0;
    state.R__prev = 0;
#endif

    state.in_buf    = src;
    state.in_size   = (unsigned) (*srcsize);
    state.in_offset = 0;

    state.out_buf     = tgt;
    state.out_size    = (unsigned) (*tgtsize);
    state.out_offset  = HDRSIZE;
    state.R__window_size = 0L;

    if (0 != R__bi_init(&state) ) return;       /* initialize bit routines */
    state.t_state = R__get_thread_tree_state();
    if (0 != R__ct_init(state.t_state,&att, &method)) return; /* initialize tree routines */
    if (0 != R__lm_init(&state, gCompressionLevel, &flags)) return; /* initialize compression */
    R__Deflate(&state,&state.error_flag);                  /* compress data */
    if (state.error_flag != 0) return;

    tgt[0] = 'C';               /* Signature 'C'-Chernyaev, 'S'-Smirnov */
    tgt[1] = 'S';
    tgt[2] = (char) method;

    state.out_size  = state.out_offset - HDRSIZE;         /* compressed size */
    tgt[3] = (char)(state.out_size & 0xff);
    tgt[4] = (char)((state.out_size >> 8) & 0xff);
    tgt[5] = (char)((state.out_size >> 16) & 0xff);

    tgt[6] = (char)(state.in_size & 0xff);         /* decompressed size */
    tgt[7] = (char)((state.in_size >> 8) & 0xff);
    tgt[8] = (char)((state.in_size >> 16) & 0xff);

    *irep     = state.out_offset;
    return;
}

/**
 * Compress buffer contents using the venerable zlib algorithm.
 */
static void R__zipZLIB(int cxlevel, int *srcsize, char *src, int *tgtsize, char *tgt, int *irep)
{
  int err;
  int method   = Z_DEFLATED;

    z_stream stream;
    //Don't use the globals but want name similar to help see similarities in code
    unsigned l_in_size, l_out_size;
    *irep = 0;

    /* error_flag   = 0; */
    if (*tgtsize <= 0) {
       R__error("target buffer too small");
       return;
    }
    if (*srcsize > 0xffffff) {
       R__error("source buffer too big");
       return;
    }

    stream.next_in   = (Bytef*)src;
    stream.avail_in  = (uInt)(*srcsize);

    stream.next_out  = (Bytef*)(&tgt[HDRSIZE]);
    stream.avail_out = (uInt)(*tgtsize);

    stream.zalloc    = (alloc_func)0;
    stream.zfree     = (free_func)0;
    stream.opaque    = (voidpf)0;

    if (cxlevel > 9) cxlevel = 9;
    err = deflateInit(&stream, cxlevel);
    if (err != Z_OK) {
       printf("error %d in deflateInit (zlib)\n",err);
       return;
    }

    while ((err = deflate(&stream, Z_FINISH)) != Z_STREAM_END) {
       if (err != Z_OK) {
          deflateEnd(&stream);
          return;
       }
    }

    err = deflateEnd(&stream);

    tgt[0] = 'Z';               /* Signature ZLib */
    tgt[1] = 'L';
    tgt[2] = (char) method;

    l_in_size   = (unsigned) (*srcsize);
    l_out_size  = stream.total_out;             /* compressed size */
    tgt[3] = (char)(l_out_size & 0xff);
    tgt[4] = (char)((l_out_size >> 8) & 0xff);
    tgt[5] = (char)((l_out_size >> 16) & 0xff);

    tgt[6] = (char)(l_in_size & 0xff);         /* decompressed size */
    tgt[7] = (char)((l_in_size >> 8) & 0xff);
    tgt[8] = (char)((l_in_size >> 16) & 0xff);

    *irep = stream.total_out + HDRSIZE;
    return;
}


void R__zip(int cxlevel, int *srcsize, char *src, int *tgtsize, char *tgt, int *irep) {
   R__zipMultipleAlgorithm(cxlevel, srcsize, src, tgtsize, tgt, irep,
                           ROOT::ECompressionAlgorithm::kUseGlobalCompressionSetting);
}
