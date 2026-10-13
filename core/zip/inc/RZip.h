// @(#)root/zip:$Id$
// Author: Sergey Linev   7 July 2014

/*************************************************************************
 * Copyright (C) 1995-2014, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#include "Compression.h"

/**
 * These are definitions of various free functions for the C-style compression routines in ROOT.
 */

#ifndef ROOT_RZip
#define ROOT_RZip

namespace ROOT::Internal {

constexpr int kZipHeaderSize = 9; ///< Number of bytes of the ROOT compression block header

} // namespace ROOT::Internal

extern "C" unsigned long R__crc32(unsigned long crc, const unsigned char *buf, unsigned int len);

extern "C" unsigned long R__memcompress(char *tgt, unsigned long tgtsize, const char *src, unsigned long srcsize);

extern "C" void R__zipMultipleAlgorithm(int cxlevel, int *srcsize, const char *src, int *tgtsize, char *tgt, int *irep,
                                        ROOT::RCompressionSetting::EAlgorithm::EValues algorithm);

extern "C" ROOT::RCompressionSetting::EAlgorithm::EValues
R__getCompressionAlgorithm(const unsigned char *buf, size_t bufsize);

extern "C" void R__unzip(int *srcsize, const unsigned char *src, int *tgtsize, unsigned char *tgt, int *irep);

extern "C" int R__unzip_header(int *srcsize, const unsigned char *src, int *tgtsize);

enum { kMAXZIPBUF = 0xffffff }; // 16 MB

namespace ROOT::Internal {

/// Size of the blocks in which a buffer of `nbytes` > 0 bytes is compressed. The buffer is cut into the fewest
/// blocks of at most kMAXZIPBUF bytes, `1 + (nbytes - 1) / kMAXZIPBUF`, all of this size but the last, which is
/// smaller by less than the number of blocks. Full blocks followed by the remainder would end a buffer just above a
/// multiple of kMAXZIPBUF in a block of a few bytes, which cannot be compressed, so that the whole buffer would be
/// stored uncompressed.
constexpr int GetZipBlockSize(long long nbytes)
{
   const long long nblocks = 1 + (nbytes - 1) / kMAXZIPBUF;
   return static_cast<int>((nbytes + nblocks - 1) / nblocks);
}

} // namespace ROOT::Internal

#endif
