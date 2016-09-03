// @(#)root/core:$Id$
// Author: Manuel Schiller, Paul Seyfert 26/05/16

/*************************************************************************
 * Copyright (C) 1995-2016, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "zlib.h"

#ifdef BUILTIN_LZ4
#include "builtin/lz4.h"
#else
#include "lz4.h"
#endif

#ifndef NULL
#define NULL 0L
#endif


typedef char              boolean;
typedef unsigned char     uch;  /* code assumes unsigned bytes; these type-  */
typedef unsigned short    ush;  /*  defs replace byte/UWORD/ULONG (which are */
typedef unsigned long     ulg;  /*  predefined on some systems) & match zip  */

extern void R__error(const char *msg);
#define HDRSIZE 9

void R__zipLZ4(int cxlevel, int *srcsize, char *src, int *tgtsize, char *tgt, int *irep)
{
   int osz = *tgtsize, minosz;
   int ibufsz = *srcsize;
   unsigned level = (cxlevel ? 1 : 0);
   unsigned long adler = 0;
   /* check source buffer size */
   if (ibufsz > 0xffffff) {
      R__error("source buffer too large");
      *irep = 0;
      return;
   }
   /* calculate the buffer size needed for safe in place compression */
   minosz = LZ4_compressBound(ibufsz);
   if (0 == level) minosz = ibufsz;
   minosz += HDRSIZE + 4; /* header plus check sum */
   /* check buffer sizes */
   if (osz <= HDRSIZE + 4) {
      R__error("target buffer too small");
      *irep = 0;
      return;
   }
   /* init header */
   tgt[0] = 'L';
   tgt[1] = '4';
   /* compress with specified level and algorithm */
   if (level > 0) {
      uch *obuf = (uch*)tgt + HDRSIZE;
#ifdef BUILTIN_LZ4
      int csz = LZ4_compress((const char *) src, (char *) obuf, ibufsz);
#else
      int csz = LZ4_compress_default((const char *) src, (char *) obuf, ibufsz, osz);
#endif
      /* check compression ratio */
      if (csz < ibufsz && 0 != level) {
         tgt[2] = 1;
         *irep = csz + HDRSIZE + 4;
      } else {
         /* uncompressible, try to store uncompressed below */
         level = 0;
      }
   }
   /* check if we are to store uncompressed */
   if (0 == level) {
      /* check for sufficient space */
      minosz = ibufsz + HDRSIZE + 4;
      if (osz < minosz) {
         R__error("target buffer too small");
         *irep = 0;
         return;
      }
      *irep = minosz;
      /* copy to output buffer (move, buffers might overlap) */
      memmove(tgt + HDRSIZE, src, ibufsz);
      tgt[2] = 0; /* store uncompressed */
   }
   /* fill in sizes */
   osz = *irep - HDRSIZE;
   tgt[3] = (char)(osz & 0xff);        /* compressed size */
   tgt[4] = (char)((osz >> 8) & 0xff);
   tgt[5] = (char)((osz >> 16) & 0xff);

   tgt[6] = (char)(ibufsz & 0xff);        /* decompressed size */
   tgt[7] = (char)((ibufsz >> 8) & 0xff);
   tgt[8] = (char)((ibufsz >> 16) & 0xff);
   /* calculate checksum */
   adler = adler32(
                adler32(0, NULL, 0), (uch*)tgt + HDRSIZE, osz - 4);
   tgt += *irep - 4;
   tgt[0] = (char)(adler & 0xff);
   tgt[1] = (char)((adler >> 8) & 0xff);
   tgt[2] = (char)((adler >> 16) & 0xff);
   tgt[3] = (char)((adler >> 24) & 0xff);

   return;
}

int R__unzipLZ4(uch *ibufptr, long ibufsz,
                uch *obufptr, long *obufsz, uch method)
{
   if (ibufsz < 4) {
      return -1;
   }
   /* TODO reenable checksum check */
   if (0) {
      /* check adler32 checksum */
      uch *p = ibufptr + (ibufsz - 4);
      unsigned long adler = ((unsigned long) p[0]) | ((unsigned long) p[1] << 8) |
                            ((unsigned long) p[2] << 16) | ((unsigned long) p[3] << 24);
      if (adler != adler32(adler32(0, NULL, 0), ibufptr, ibufsz - 4)) {
         /* corrupt compressed data */
         return -2;
      }
   }
   int osz;
   switch (method) {
      case 0: /* just store the uncompressed data */
         if (*obufsz != ibufsz - 4) return -3;
         if (ibufptr != obufptr)
            memmove(obufptr, ibufptr, ibufsz - 4);
         break;
      case 1: /* LZ4 */
#ifdef BUILTIN_LZ4
         osz = LZ4_uncompress_unknownOutputSize((const char *) ibufptr, (char *)obufptr, ibufsz - 4, *obufsz);
         if (osz != *obufsz) return -4;
         /* TODO: use target size from header */
#else
         osz = LZ4_decompress_fast((const char *) ibufptr, (char *)obufptr, *obufsz);
         if (osz != *obufsz) return -5;
#endif
         break;
      default:
         /* unknown method */
         return -6;
   }
   return 0;
}


