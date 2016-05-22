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


int R__ZopfliCompress(ZopfliOptions* zpfopts, ZopfliFormat zpftype,
        uch* src, size_t srcsize, char* target, size_t* dstsz)
{
  uch* compression_target = 0;
  size_t compression_size = 0;
  unsigned long adler32;
  unsigned int obufs;
  unsigned int osz;
  unsigned int* obufsz;
  char* obufptr;

  obufs = *dstsz;
  obufsz = &obufs;
  (target)[0] = 'Z';
  (target)[1] = 'P';
  if (ZOPFLI_FORMAT_ZLIB == zpftype) (target)[2] = 'Z';
  if (ZOPFLI_FORMAT_GZIP == zpftype) (target)[2] = 'G';
  if (ZOPFLI_FORMAT_DEFLATE == zpftype) (target)[2] = 'D';
  ZopfliCompress(zpfopts, zpftype, src, srcsize, &compression_target, &compression_size);
  if (compression_size > srcsize) {
    free(compression_target);
    if (*dstsz < srcsize + HDRSIZE + 4) {
      R__error("could not leave uncompressed");
      return -1;
    }
    memmove(target + HDRSIZE,src,srcsize);
    target[2]=0;
    compression_size = srcsize;
  } else {
    if (*dstsz < compression_size + HDRSIZE + 4) {
      /* this is actually caught */
      R__error("could not compress");
      free(compression_target);
      return -1;
    }
    memmove(target + HDRSIZE,compression_target,compression_size);
    free(compression_target);
  }

  /* does all this make sense? */
  *dstsz = compression_size + HDRSIZE + 4;
  *obufsz = compression_size + HDRSIZE + 4;
  osz = *obufsz - HDRSIZE;
  (target)[3] = (char)(((osz) >> 0) & 0xff);
  (target)[4] = (char)(((osz) >> 8) & 0xff);
  (target)[5] = (char)(((osz) >>16) & 0xff);
  (target)[6] = (char)(((srcsize) >> 0) & 0xff);
  (target)[7] = (char)(((srcsize) >> 8) & 0xff);
  (target)[8] = (char)(((srcsize) >>16) & 0xff);
  /* calculate checksum */
  adler32 = lzo_adler32(
      lzo_adler32(0, NULL,0), (target) + HDRSIZE, osz - 4);
  obufptr = target;
  obufptr += *obufsz - 4;
  obufptr[0] = (char) (adler32 & 0xff);
  obufptr[1] = (char) ((adler32 >> 8) & 0xff);
  obufptr[2] = (char) ((adler32 >> 16) & 0xff);
  obufptr[3] = (char) ((adler32 >> 24) & 0xff);

  return 0;

}

void R__zipZOPFLI(int cxlevel, int *srcsize, char *src, int *tgtsize, char *tgt, int *irep)
{
  size_t dstsz = *tgtsize;
  ZopfliFormat zpftype;
  ZopfliOptions zpfopts;
  ZopfliInitOptions(&zpfopts);
  zpfopts.numiterations = cxlevel;
  zpftype = ZOPFLI_FORMAT_ZLIB; /* also possible GZIP or DEFLATE */
  if (R__ZopfliCompress( &zpfopts, zpftype, (uch*) src, *srcsize, tgt, &dstsz)) {
    *irep = 0;
  } else {
    *irep = dstsz;
  }
  *tgtsize = dstsz;
}


