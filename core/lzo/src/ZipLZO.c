#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lzo/lzoutil.h"
#include "lzo/lzo1.h"
#include "lzo/lzo1a.h"
#include "lzo/lzo1b.h"
#include "lzo/lzo1c.h"
#include "lzo/lzo1f.h"
#include "lzo/lzo1x.h"
#include "lzo/lzo1y.h"
#include "lzo/lzo1z.h"
#include "lzo/lzo2a.h"

#ifdef WIN32
#define __STDC__
#endif
#ifdef __MWERKS__
#define __STDC__
#endif

#ifndef NULL
#define NULL 0L
#endif

#if 0
#define PKZIP_BUG_WORKAROUND    /* PKZIP 1.93a problem--live with it */
#endif

/*
    inflate.h must supply the uch slide[WSIZE] array and the NEXTBYTE,
    FLUSH() and memzero macros.  If the window size is not 32K, it
    should also define WSIZE.  If INFMOD is defined, it can include
    compiled functions to support the NEXTBYTE and/or FLUSH() macros.
    There are defaults for NEXTBYTE and FLUSH() below for use as
    examples of what those functions need to do.  Normally, you would
    also want FLUSH() to compute a crc on the data.  inflate.h also
    needs to provide these typedefs:

        typedef unsigned char uch;
        typedef unsigned short ush;
        typedef unsigned long ulg;

    This module uses the external functions malloc() and free() (and
    probably memset() or bzero() in the memzero() macro).  Their
    prototypes are normally found in <string.h> and <stdlib.h>.
 */
typedef char              boolean;
typedef unsigned char     uch;  /* code assumes unsigned bytes; these type-  */
typedef unsigned short    ush;  /*  defs replace byte/UWORD/ULONG (which are */
typedef unsigned long     ulg;  /*  predefined on some systems) & match zip  */

/* Function prototypes */
#ifndef OF
#  ifdef __STDC__
#    define OF(a) a
#  else /* !__STDC__ */
#    define OF(a) ()
#  endif /* ?__STDC__ */
#endif
extern void R__error(const char *msg);
#define HDRSIZE 9


/***********************************************************************
 *
 * begin liblzo related routines and definitions
 *
 **********************************************************************/
static int R__lzo_init()
{
  static volatile int R__lzo_inited = 0;
  if (R__lzo_inited) return 1;
  return (R__lzo_inited = (lzo_init() == LZO_E_OK));
}

int R__unzipLZO(uch* ibufptr, long ibufsz,
    uch* obufptr, long* obufsz, uch method)
{
  lzo_uint osz = *obufsz;
  if (ibufsz < 4) {
    return -1;
  }
  /* initialise liblzo */
  if (!R__lzo_init()) return -1;
  {
    /* check adler32 checksum */
    uch *p = ibufptr + (ibufsz - 4);
    unsigned long adler = ((unsigned long) p[0]) | ((unsigned long) p[1] << 8) |
      ((unsigned long) p[2] << 16) | ((unsigned long) p[3] << 24);
    if (adler != lzo_adler32(lzo_adler32(0, NULL, 0), ibufptr, ibufsz - 4)) {
      /* corrupt compressed data */
      return -1;
    }
  }
  switch (method) {
    case 0: /* just store the uncompressed data */
      if (*obufsz != ibufsz - 4) return -1;
      if (ibufptr != obufptr)
        lzo_memmove(obufptr, ibufptr, ibufsz - 4);
      break;
    case 1: /* LZO1 */
      if (LZO_E_OK != lzo1_decompress(ibufptr, ibufsz - 4,
            obufptr, &osz, NULL) || ((unsigned long) *obufsz != osz))
        return -1;
      break;
    case 2: /* LZO1A */
      if (LZO_E_OK != lzo1a_decompress(ibufptr, ibufsz - 4,
            obufptr, &osz, NULL) || ((unsigned long) *obufsz != osz))
        return -1;
      break;
    case 3: /* LZO1B */
      if (LZO_E_OK != lzo1b_decompress_safe(ibufptr, ibufsz - 4,
            obufptr, &osz, NULL) || ((unsigned long) *obufsz != osz))
        return -1;
      break;
    case 4: /* LZO1C */
      if (LZO_E_OK != lzo1c_decompress_safe(ibufptr, ibufsz - 4,
            obufptr, &osz, NULL) || ((unsigned long) *obufsz != osz))
        return -1;
      break;
    case 5: /* LZO1F */
      if (LZO_E_OK != lzo1f_decompress_safe(ibufptr, ibufsz - 4,
            obufptr, &osz, NULL) || ((unsigned long) *obufsz != osz))
        return -1;
      break;
    case 6: /* LZO1X */
      if (LZO_E_OK != lzo1x_decompress_safe(ibufptr, ibufsz - 4,
            obufptr, &osz, NULL) || ((unsigned long) *obufsz != osz))
        return -1;
      break;
    case 7: /* LZO1Y */
      if (LZO_E_OK != lzo1y_decompress_safe(ibufptr, ibufsz - 4,
            obufptr, &osz, NULL) || ((unsigned long) *obufsz != osz))
        return -1;
      break;
    case 8: /* LZO1Z */
      if (LZO_E_OK != lzo1z_decompress_safe(ibufptr, ibufsz - 4,
            obufptr, &osz, NULL) || ((unsigned long) *obufsz != osz))
        return -1;
      break;
    case 9: /* LZO2A */
      if (LZO_E_OK != lzo2a_decompress_safe(ibufptr, ibufsz - 4,
            obufptr, &osz, NULL) || ((unsigned long) *obufsz != osz))
        return -1;
      break;
    default:
      /* unknown method */
      return -1;
  }
  return 0;
}

/* definition of struct to map compression level to algorithms and settings */
struct R__lzo_tbl_t {
  int method;                        /* method code to be written to file */
  lzo_compress_t compress;        /* ptr to compression routine */
  unsigned long wkspsz;        /* size of required workspace */
  lzo_optimize_t optimize;        /* ptr to optimize routine */
};

static struct R__lzo_tbl_t R__lzo_compr_tbl[9][11] = {
  { /* idx 0: LZO1X */
    { 6, lzo1x_1_11_compress, LZO1X_1_11_MEM_COMPRESS, lzo1x_optimize },
    { 6, lzo1x_1_12_compress, LZO1X_1_12_MEM_COMPRESS, lzo1x_optimize },
    { 6, lzo1x_1_compress, LZO1X_1_MEM_COMPRESS, lzo1x_optimize },
    { 6, lzo1x_1_compress, LZO1X_1_MEM_COMPRESS, lzo1x_optimize },
    { 6, lzo1x_1_15_compress, LZO1X_1_15_MEM_COMPRESS, lzo1x_optimize },
    { 6, lzo1x_1_15_compress, LZO1X_1_15_MEM_COMPRESS, lzo1x_optimize },
    { 6, lzo1x_999_compress, LZO1X_999_MEM_COMPRESS, lzo1x_optimize },
    { 6, lzo1x_999_compress, LZO1X_999_MEM_COMPRESS, lzo1x_optimize },
    { 6, lzo1x_999_compress, LZO1X_999_MEM_COMPRESS, lzo1x_optimize },
    { 6, lzo1x_999_compress, LZO1X_999_MEM_COMPRESS, lzo1x_optimize },
    { 6, lzo1x_999_compress, LZO1X_999_MEM_COMPRESS, lzo1x_optimize },
  },
  { /* idx 1: LZO1 */
    { 1, lzo1_compress, LZO1_MEM_COMPRESS, NULL },
    { 1, lzo1_compress, LZO1_MEM_COMPRESS, NULL },
    { 1, lzo1_compress, LZO1_MEM_COMPRESS, NULL },
    { 1, lzo1_compress, LZO1_MEM_COMPRESS, NULL },
    { 1, lzo1_compress, LZO1_MEM_COMPRESS, NULL },
    { 1, lzo1_compress, LZO1_MEM_COMPRESS, NULL },
    { 1, lzo1_99_compress, LZO1_99_MEM_COMPRESS, NULL },
    { 1, lzo1_99_compress, LZO1_99_MEM_COMPRESS, NULL },
    { 1, lzo1_99_compress, LZO1_99_MEM_COMPRESS, NULL },
    { 1, lzo1_99_compress, LZO1_99_MEM_COMPRESS, NULL },
    { 1, lzo1_99_compress, LZO1_99_MEM_COMPRESS, NULL },
  },
  { /* idx 2: LZO1A */
    { 2, lzo1a_compress, LZO1A_MEM_COMPRESS, NULL },
    { 2, lzo1a_compress, LZO1A_MEM_COMPRESS, NULL },
    { 2, lzo1a_compress, LZO1A_MEM_COMPRESS, NULL },
    { 2, lzo1a_compress, LZO1A_MEM_COMPRESS, NULL },
    { 2, lzo1a_compress, LZO1A_MEM_COMPRESS, NULL },
    { 2, lzo1a_compress, LZO1A_MEM_COMPRESS, NULL },
    { 2, lzo1a_99_compress, LZO1A_99_MEM_COMPRESS, NULL },
    { 2, lzo1a_99_compress, LZO1A_99_MEM_COMPRESS, NULL },
    { 2, lzo1a_99_compress, LZO1A_99_MEM_COMPRESS, NULL },
    { 2, lzo1a_99_compress, LZO1A_99_MEM_COMPRESS, NULL },
    { 2, lzo1a_99_compress, LZO1A_99_MEM_COMPRESS, NULL },
  },
  { /* idx 3: LZO1B */
    { 3, lzo1b_1_compress, LZO1B_MEM_COMPRESS, NULL },
    { 3, lzo1b_2_compress, LZO1B_MEM_COMPRESS, NULL },
    { 3, lzo1b_3_compress, LZO1B_MEM_COMPRESS, NULL },
    { 3, lzo1b_4_compress, LZO1B_MEM_COMPRESS, NULL },
    { 3, lzo1b_5_compress, LZO1B_MEM_COMPRESS, NULL },
    { 3, lzo1b_6_compress, LZO1B_MEM_COMPRESS, NULL },
    { 3, lzo1b_7_compress, LZO1B_MEM_COMPRESS, NULL },
    { 3, lzo1b_8_compress, LZO1B_MEM_COMPRESS, NULL },
    { 3, lzo1b_9_compress, LZO1B_MEM_COMPRESS, NULL },
    { 3, lzo1b_99_compress, LZO1B_99_MEM_COMPRESS, NULL },
    { 3, lzo1b_999_compress, LZO1B_999_MEM_COMPRESS, NULL },
  },
  { /* idx 4: LZO1C */
    { 4, lzo1c_1_compress, LZO1C_MEM_COMPRESS, NULL },
    { 4, lzo1c_2_compress, LZO1C_MEM_COMPRESS, NULL },
    { 4, lzo1c_3_compress, LZO1C_MEM_COMPRESS, NULL },
    { 4, lzo1c_4_compress, LZO1C_MEM_COMPRESS, NULL },
    { 4, lzo1c_5_compress, LZO1C_MEM_COMPRESS, NULL },
    { 4, lzo1c_6_compress, LZO1C_MEM_COMPRESS, NULL },
    { 4, lzo1c_7_compress, LZO1C_MEM_COMPRESS, NULL },
    { 4, lzo1c_8_compress, LZO1C_MEM_COMPRESS, NULL },
    { 4, lzo1c_9_compress, LZO1C_MEM_COMPRESS, NULL },
    { 4, lzo1c_99_compress, LZO1C_99_MEM_COMPRESS, NULL },
    { 4, lzo1c_999_compress, LZO1C_999_MEM_COMPRESS, NULL },
  },
  { /* idx 5: LZO1F */
    { 5, lzo1f_1_compress, LZO1F_MEM_COMPRESS, NULL },
    { 5, lzo1f_1_compress, LZO1F_MEM_COMPRESS, NULL },
    { 5, lzo1f_1_compress, LZO1F_MEM_COMPRESS, NULL },
    { 5, lzo1f_1_compress, LZO1F_MEM_COMPRESS, NULL },
    { 5, lzo1f_1_compress, LZO1F_MEM_COMPRESS, NULL },
    { 5, lzo1f_1_compress, LZO1F_MEM_COMPRESS, NULL },
    { 5, lzo1f_999_compress, LZO1F_999_MEM_COMPRESS, NULL },
    { 5, lzo1f_999_compress, LZO1F_999_MEM_COMPRESS, NULL },
    { 5, lzo1f_999_compress, LZO1F_999_MEM_COMPRESS, NULL },
    { 5, lzo1f_999_compress, LZO1F_999_MEM_COMPRESS, NULL },
    { 5, lzo1f_999_compress, LZO1F_999_MEM_COMPRESS, NULL },
  },
  { /* idx 6: LZO1Y */
    { 7, lzo1y_1_compress, LZO1Y_MEM_COMPRESS, lzo1y_optimize },
    { 7, lzo1y_1_compress, LZO1Y_MEM_COMPRESS, lzo1y_optimize },
    { 7, lzo1y_1_compress, LZO1Y_MEM_COMPRESS, lzo1y_optimize },
    { 7, lzo1y_1_compress, LZO1Y_MEM_COMPRESS, lzo1y_optimize },
    { 7, lzo1y_1_compress, LZO1Y_MEM_COMPRESS, lzo1y_optimize },
    { 7, lzo1y_1_compress, LZO1Y_MEM_COMPRESS, lzo1y_optimize },
    { 7, lzo1y_999_compress, LZO1Y_999_MEM_COMPRESS, lzo1y_optimize },
    { 7, lzo1y_999_compress, LZO1Y_999_MEM_COMPRESS, lzo1y_optimize },
    { 7, lzo1y_999_compress, LZO1Y_999_MEM_COMPRESS, lzo1y_optimize },
    { 7, lzo1y_999_compress, LZO1Y_999_MEM_COMPRESS, lzo1y_optimize },
    { 7, lzo1y_999_compress, LZO1Y_999_MEM_COMPRESS, lzo1y_optimize },
  },
  { /* idx 7: LZO1Z */
    { 8, lzo1z_999_compress, LZO1Z_999_MEM_COMPRESS, NULL },
    { 8, lzo1z_999_compress, LZO1Z_999_MEM_COMPRESS, NULL },
    { 8, lzo1z_999_compress, LZO1Z_999_MEM_COMPRESS, NULL },
    { 8, lzo1z_999_compress, LZO1Z_999_MEM_COMPRESS, NULL },
    { 8, lzo1z_999_compress, LZO1Z_999_MEM_COMPRESS, NULL },
    { 8, lzo1z_999_compress, LZO1Z_999_MEM_COMPRESS, NULL },
    { 8, lzo1z_999_compress, LZO1Z_999_MEM_COMPRESS, NULL },
    { 8, lzo1z_999_compress, LZO1Z_999_MEM_COMPRESS, NULL },
    { 8, lzo1z_999_compress, LZO1Z_999_MEM_COMPRESS, NULL },
    { 8, lzo1z_999_compress, LZO1Z_999_MEM_COMPRESS, NULL },
    { 8, lzo1z_999_compress, LZO1Z_999_MEM_COMPRESS, NULL },
  },
  { /* idx 8: LZO2A */
    { 9, lzo2a_999_compress, LZO2A_999_MEM_COMPRESS, NULL },
    { 9, lzo2a_999_compress, LZO2A_999_MEM_COMPRESS, NULL },
    { 9, lzo2a_999_compress, LZO2A_999_MEM_COMPRESS, NULL },
    { 9, lzo2a_999_compress, LZO2A_999_MEM_COMPRESS, NULL },
    { 9, lzo2a_999_compress, LZO2A_999_MEM_COMPRESS, NULL },
    { 9, lzo2a_999_compress, LZO2A_999_MEM_COMPRESS, NULL },
    { 9, lzo2a_999_compress, LZO2A_999_MEM_COMPRESS, NULL },
    { 9, lzo2a_999_compress, LZO2A_999_MEM_COMPRESS, NULL },
    { 9, lzo2a_999_compress, LZO2A_999_MEM_COMPRESS, NULL },
    { 9, lzo2a_999_compress, LZO2A_999_MEM_COMPRESS, NULL },
    { 9, lzo2a_999_compress, LZO2A_999_MEM_COMPRESS, NULL },
  },
};

void R__zipLZO(int cxlevel, int *srcsize, char *src, int *tgtsize, char *tgt, int *irep)
{
  lzo_uint ibufsz = *srcsize;
  lzo_uint osz = *irep, minosz;
  unsigned long adler32 = 0;
  int level = cxlevel & 0xf;
  int alg = (cxlevel >> 4) & 0xf;
  int opt = cxlevel & 0x100;
  *irep = 0;
  if (level > 0xb || alg > 0x8 || cxlevel > 0x1ff) {
    *irep = 0;
    return;
  }
  if (0 == level) alg = opt = 0;
  /* calculate the buffer size needed for safe in place compression */
  minosz = ibufsz + (ibufsz / 16) + 64 + 3;
  if (8 == alg) minosz = ibufsz + (ibufsz / 8) + 128 + 3;
  if (0 == level) minosz = ibufsz;
  minosz += HDRSIZE + 4; /* header plus check sum */
  /* check buffer sizes */
  if (osz <= HDRSIZE + 4) {
    R__error("target buffer too small");
    *irep = 0;
    return;
  }
  if (ibufsz > 0xffffff) {
    R__error("source buffer too large");
    *irep = 0;
    return;
  }
  /* init header */
  tgt[0] = 'L';
  tgt[1] = 'Z';
  /* compress with specified level and algorithm */
  if (level > 0) {
    struct R__lzo_tbl_t *algp = &R__lzo_compr_tbl[alg][level - 1];
    uch* obuf = tgt + HDRSIZE;
    uch* wksp = NULL;
    lzo_uint csz = 0;
    /* initialise liblzo */
    if (!R__lzo_init()) {
      *irep = 0;
      return;
    }
    /* allocate workspace and safe temp buffer (if needed) */
    if (minosz > osz) {
      wksp = (uch*) lzo_malloc(algp->wkspsz + minosz - HDRSIZE - 4);
      obuf = wksp + algp->wkspsz;
    } else {
      wksp = (uch*) lzo_malloc(algp->wkspsz);
    }
    if (NULL == wksp) {
      R__error("out of memory");
      *irep = 0;
      return;
    }
    /* compress */
    if (LZO_E_OK != algp->compress(src, ibufsz, obuf, &csz, wksp)) {
      /* something is wrong, try to store uncompressed below */
      alg = level = opt = 0;
      R__error("liblzo: unable to compress, trying to store as is");
    } else {
      /* compression ok, check if we need to optimize */
      if (opt && algp->optimize) {
        lzo_uint ucsz = ibufsz;
        if (LZO_E_OK != algp->optimize(obuf, csz, src, &ucsz, NULL) ||
            ibufsz != ucsz) {
          /* something is wrong, try to store uncompressed below */
          alg = level = opt = 0;
          R__error("liblzo: unable to optimize, trying to store as is");
        }
      }

      /* check compression ratio */
      if (csz < ibufsz && 0 != level) {
        /* check if we need to copy from temp to final buffer */
        if (obuf != tgt + HDRSIZE) {
          /* check for sufficient space and copy */
          minosz = csz + HDRSIZE + 4;
          if (osz < minosz) {
            /* not enough space - try to store */
            alg = level = opt = 0;
          } else {
            lzo_memcpy(tgt + HDRSIZE, obuf, csz);
            tgt[2] = algp->method;
            *irep = csz + HDRSIZE + 4;
          }
        } else {
          tgt[2] = algp->method;
          *irep = csz + HDRSIZE + 4;
        }
      } else {
        /* uncompressible, try to store uncompressed below */
        alg = level = opt = 0;
      }
    }
    lzo_free(wksp);
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
    lzo_memmove(tgt + HDRSIZE, src, ibufsz);
    tgt[2] = 0; /* store uncompressed */
  };
  /* fill in sizes */
  osz = *irep - HDRSIZE;
  tgt[3] = (char)(osz & 0xff);        /* compressed size */
  tgt[4] = (char)((osz >> 8) & 0xff);
  tgt[5] = (char)((osz >> 16) & 0xff);

  tgt[6] = (char)(ibufsz & 0xff);        /* decompressed size */
  tgt[7] = (char)((ibufsz >> 8) & 0xff);
  tgt[8] = (char)((ibufsz >> 16) & 0xff);
  /* calculate checksum */
  adler32 = lzo_adler32(
          lzo_adler32(0, NULL,0), tgt + HDRSIZE, osz - 4);
  tgt += *irep - 4;
  tgt[0] = (char) (adler32 & 0xff);
  tgt[1] = (char) ((adler32 >> 8) & 0xff);
  tgt[2] = (char) ((adler32 >> 16) & 0xff);
  tgt[3] = (char) ((adler32 >> 24) & 0xff);

  return;
}
