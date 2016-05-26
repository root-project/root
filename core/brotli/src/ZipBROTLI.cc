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
#include <stddef.h>
#include <string.h>

#include "zlib.h"


#include "enc/encode.h"
#include "dec/decode.h"

typedef char              boolean;
typedef unsigned char     uch;  /* code assumes unsigned bytes; these type-  */
typedef unsigned short    ush;  /*  defs replace byte/UWORD/ULONG (which are */
typedef unsigned long     ulg;  /*  predefined on some systems) & match zip  */

extern "C" void R__error(const char *msg);
#define HDRSIZE 9

extern "C" int R__BrotliCompress(int cxlevel , uch *src, size_t srcsize, uch *target, size_t *dstsz);

extern "C" int R__unzipBROTLI(uch *ibufptr, long ibufsz, uch *obufptr, size_t *obufsz);

extern "C" void R__zipBROTLI(int cxlevel, int *srcsize, char *src, int *tgtsize, char *tgt, int *irep)
{
   size_t dstsz = *tgtsize;
   int retcode = R__BrotliCompress(cxlevel, (uch *) src, *srcsize, (uch *) tgt, &dstsz);
   if (retcode) *irep = 0;
   else *irep = dstsz;
   *tgtsize = dstsz;
   return;
}

int R__BrotliCompress(int cxlevel , uch *src, size_t srcsize, uch *target, size_t *dstsz)
{
   brotli::BrotliParams params;
   size_t compression_size = *dstsz;
   unsigned long adler;
   unsigned int osz;
   uch *obufptr;
   int status;

   params.quality = cxlevel;
   (target)[0] = 'B';
   (target)[1] = 'R';
   (target)[2] = 'O';
   status = BrotliCompressBuffer(params, srcsize, src, &compression_size, target + HDRSIZE);
   if (status == 0) {
      return -1;
      if (*dstsz < srcsize + HDRSIZE + 4) {
         R__error("could not leave uncompressed");
         return -1;
      }
      memmove(target + HDRSIZE, src, srcsize);
      target[2] = 0;
      compression_size = srcsize;
   } else {
      if (*dstsz < compression_size + HDRSIZE + 4) {
         /* this is actually caught */
         R__error("could not compress");
         return -1;
      }
   }

   /* does all this make sense? */
   *dstsz = compression_size + HDRSIZE + 4;
   osz = *dstsz - HDRSIZE;
   (target)[3] = (char)(((osz) >> 0) & 0xff);
   (target)[4] = (char)(((osz) >> 8) & 0xff);
   (target)[5] = (char)(((osz) >> 16) & 0xff);
   (target)[6] = (char)(((srcsize) >> 0) & 0xff);
   (target)[7] = (char)(((srcsize) >> 8) & 0xff);
   (target)[8] = (char)(((srcsize) >> 16) & 0xff);
   /* calculate checksum */
   adler = adler32(
                adler32(0, NULL, 0), (target) + HDRSIZE, osz - 4);
   obufptr = target;
   obufptr += *dstsz - 4;
   obufptr[0] = (char)(adler & 0xff);
   obufptr[1] = (char)((adler >> 8) & 0xff);
   obufptr[2] = (char)((adler >> 16) & 0xff);
   obufptr[3] = (char)((adler >> 24) & 0xff);

   /*printf("header write %d\t%d\n",srcsize,osz);*/
   return 0;

}

int R__unzipBROTLI(uch *ibufptr, long ibufsz, uch *obufptr, size_t *obufsz)
{
   int status;
   if (ibufsz < 4) {
      return -1;
   }
   if (false) {
      /* check adler32 checksum */
      uch *p = ibufptr + (ibufsz - 4);
      unsigned long adler = ((unsigned long) p[0]) | ((unsigned long) p[1] << 8) |
                            ((unsigned long) p[2] << 16) | ((unsigned long) p[3] << 24);
      if (adler != adler32(adler32(0, NULL, 0), ibufptr, ibufsz - 4)) {
         /* corrupt compressed data */
         return -1;
      }
   }
   status = BrotliDecompressBuffer(ibufsz - 4, ibufptr, obufsz, obufptr);
   if (0 == status) {
      return -1;
   }
   return 0;
}

