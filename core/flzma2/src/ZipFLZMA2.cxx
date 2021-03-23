// Original Author: Oksana Shadura
/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#include "ZipFLZMA2.h"
#include "ROOT/RConfig.hxx"

#include "fast-lzma2.h"

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <iostream>

static const int kHeaderSize = 9;

void R__zipFLZMA2(int cxlevel, int *srcsize, char *src, int *tgtsize, char *tgt, int *irep)
{  
   using Ctx_ptr = std::unique_ptr<FL2_CCtx, decltype(&FL2_freeCCtx)>;
   Ctx_ptr fCtx{FL2_createCCtx(), &FL2_freeCCtx};

   *irep = 0;

   size_t retval = FL2_compressCCtx(fCtx.get(),
                                        &tgt[kHeaderSize], static_cast<size_t>(*tgtsize - kHeaderSize),
                                        src, static_cast<size_t>(*srcsize),
                                        2*cxlevel);

   *irep = static_cast<size_t>(retval + kHeaderSize);

   size_t deflate_size = retval;
   size_t inflate_size = static_cast<size_t>(*srcsize);
   tgt[0] = 'X';
   tgt[1] = 'Z';
   tgt[2] = 0;
   tgt[3] = deflate_size & 0xff;
   tgt[4] = (deflate_size >> 8) & 0xff;
   tgt[5] = (deflate_size >> 16) & 0xff;
   tgt[6] = inflate_size & 0xff;
   tgt[7] = (inflate_size >> 8) & 0xff;
   tgt[8] = (inflate_size >> 16) & 0xff;
}

// We will not use in this test case since files supposely should be decompressed with LZMA:
// src[0] != 'X' || src[1] != 'Z'
void R__unzipFLZMA2(int *srcsize, unsigned char *src, int *tgtsize, unsigned char *tgt, int *irep)
{
   using Ctx_ptr = std::unique_ptr<FL2_DCtx, decltype(&FL2_freeDCtx)>;
   Ctx_ptr fCtx{FL2_createDCtx(), &FL2_freeDCtx};
   *irep = 0;

   if (R__unlikely(src[0] != 'X' || src[1] != 'Z' || src[2] == 0)) {
      std::cerr << "R__unzipLZMAFL2: algorithm run against buffer with incorrect header (got " <<
      src[0] << src[1] << src[2] << "; expected XZ)." << std::endl;
      return;
   }

   size_t retval = FL2_decompressDCtx(fCtx.get(),
                                       (char *)tgt, static_cast<size_t>(*tgtsize),
                                       (char *)&src[kHeaderSize], static_cast<size_t>(*srcsize - kHeaderSize));

   *irep = retval;
}
