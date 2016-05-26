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

#include "zopfli/zopfli.h"

#ifndef NULL
#define NULL 0L
#endif

typedef char              boolean;
typedef unsigned char     uch;  /* code assumes unsigned bytes; these type-  */
typedef unsigned short    ush;  /*  defs replace byte/UWORD/ULONG (which are */
typedef unsigned long     ulg;  /*  predefined on some systems) & match zip  */

extern void R__error(const char *msg);
#define HDRSIZE 9

void R__zipZOPFLI(int cxlevel, int *srcsize, const char *src, int *tgtsize, char *tgt, int *irep)
{
   static ZopfliFormat zpftype;
   static ZopfliOptions zpfopts;
   ZopfliInitOptions(&zpfopts);
   zpfopts.numiterations = cxlevel;
   zpftype = ZOPFLI_FORMAT_ZLIB; /* also possible GZIP or DEFLATE */
   uch *compression_target = 0;
   size_t compression_size = 0;
   unsigned long adler;
   unsigned int osz;
   char *obufptr;

   (tgt)[0] = 'Z';
   (tgt)[1] = 'P';
   if (ZOPFLI_FORMAT_ZLIB == zpftype)(tgt)[2] = 'Z';
   if (ZOPFLI_FORMAT_GZIP == zpftype)(tgt)[2] = 'G';
   if (ZOPFLI_FORMAT_DEFLATE == zpftype)(tgt)[2] = 'D';
   ZopfliCompress(&zpfopts, zpftype, src, *srcsize, &compression_target, &compression_size);
   if (compression_size > *srcsize) {
      free(compression_target);
      if (*tgtsize < *srcsize + HDRSIZE + 4) {
         R__error("could not leave uncompressed");
         *irep = 0;
         return;
      }
      memmove(tgt + HDRSIZE, src, *srcsize);
      tgt[2] = 0;
      compression_size = *srcsize;
   } else {
      if (*tgtsize < compression_size + HDRSIZE + 4) {
         /* this is actually caught */
         R__error("could not compress");
         free(compression_target);
         *irep = 0;
         return;
      }
      memmove(tgt + HDRSIZE, compression_target, compression_size);
      free(compression_target);
   }

   /* does all this make sense? */
   *irep = compression_size + HDRSIZE + 4;
   osz = *irep - HDRSIZE;
   (tgt)[3] = (char)(((osz) >> 0) & 0xff);
   (tgt)[4] = (char)(((osz) >> 8) & 0xff);
   (tgt)[5] = (char)(((osz) >> 16) & 0xff);
   (tgt)[6] = (char)(((*srcsize) >> 0) & 0xff);
   (tgt)[7] = (char)(((*srcsize) >> 8) & 0xff);
   (tgt)[8] = (char)(((*srcsize) >> 16) & 0xff);
   /* calculate checksum */
   adler = adler32(
              adler32(0, NULL, 0), (tgt) + HDRSIZE, osz - 4);
   obufptr = tgt;
   obufptr += *irep - 4;
   obufptr[0] = (char)(adler & 0xff);
   obufptr[1] = (char)((adler >> 8) & 0xff);
   obufptr[2] = (char)((adler >> 16) & 0xff);
   obufptr[3] = (char)((adler >> 24) & 0xff);

   return;

}



