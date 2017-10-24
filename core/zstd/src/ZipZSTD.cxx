// Original Author: Brian Bockelman

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ZipZSTD.h"

#include "ROOT/RConfig.hxx"

#include "zdict.h"
#include <zstd.h>
#include <memory>

#include <iostream>

static const int kHeaderSize = 9;

void R__zipZSTD(int cxlevel, int *srcsize, char *src, int *tgtsize, char *tgt, int *irep)
{
    using Ctx_ptr = std::unique_ptr<ZSTD_CCtx, decltype(&ZSTD_freeCCtx)>;
    Ctx_ptr fCtx{ZSTD_createCCtx(), &ZSTD_freeCCtx};

    *irep = 0;
    if (R__unlikely(*tgtsize < kHeaderSize)) {
        std::cout << "Error: target buffer too small in ZSTD" << std::endl;
        return;
    }
    size_t retval = ZSTD_compressCCtx(fCtx.get(),
                                        &tgt[kHeaderSize], static_cast<size_t>(*tgtsize - kHeaderSize),
                                        src, static_cast<size_t>(*srcsize),
                                        2*cxlevel);

    if (R__unlikely(ZSTD_isError(retval))) {
        std::cout << "Error in zip ZSTD" << std::endl;
        return;
    }
    else {
        *irep = static_cast<size_t>(retval + kHeaderSize);
    }

    size_t deflate_size = retval;
    size_t inflate_size = static_cast<size_t>(*srcsize);
    tgt[0] = 'Z';
    tgt[1] = 'S';
    tgt[2] = '\1';
    tgt[3] = deflate_size & 0xff;
    tgt[4] = (deflate_size >> 8) & 0xff;
    tgt[5] = (deflate_size >> 16) & 0xff;
    tgt[6] = inflate_size & 0xff;
    tgt[7] = (inflate_size >> 8) & 0xff;
    tgt[8] = (inflate_size >> 16) & 0xff;
}

void R__unzipZSTD(int *srcsize, unsigned char *src, int *tgtsize, unsigned char *tgt, int *irep)
{
    using Ctx_ptr = std::unique_ptr<ZSTD_DCtx, decltype(&ZSTD_freeDCtx)>;
    Ctx_ptr fCtx{ZSTD_createDCtx(), &ZSTD_freeDCtx};

    if (R__unlikely(src[0] != 'Z' || src[1] != 'S')) {
      fprintf(stderr, "R__unzipZSTD: algorithm run against buffer with incorrect header (got %d%d; expected %d%d).\n",
              src[0], src[1], 'Z', 'S');
      return;
    }

    int ZSTD_version =  ZSTD_versionNumber() / (100 * 100);
    if (R__unlikely(src[2] != ZSTD_version)) {
      fprintf(stderr,
              "R__unzipZSTD: This version of ZSTD is incompatible with the on-disk version (got %d; expected %d).\n",
              src[2], ZSTD_version);
      return;
    }

    size_t retval = ZSTD_decompressDCtx(fCtx.get(),
                                        (char *)tgt, static_cast<size_t>(*tgtsize),
                                        (char *)&src[kHeaderSize], static_cast<size_t>(*srcsize - kHeaderSize));

    if (R__unlikely(ZSTD_isError(retval))) {
        std::cout << "Error in unzip ZSTD" << std::endl;
        *irep = 0;
    }
    else {
        *irep = retval;
    }
}
