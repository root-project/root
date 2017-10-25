// @(#)root/zip:$Id$
// Author: David Dagenhart   May 2011

/*************************************************************************
 * Copyright (C) 1995-2011, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "Compression.h"
#include "ROOT/Compression.hxx"

namespace ROOT {

namespace Internal {

CompressionEngine::~CompressionEngine() {}

void
CompressionEngine::WriteROOTHeader(void *buffer, const char alg[2], char version, int deflate_size, int inflate_size) {
  char *tgt = static_cast<char *>(buffer);
  tgt[0] = alg[0];
  tgt[1] = alg[1];
  tgt[2] = version;

  tgt[3] = static_cast<char>(deflate_size & 0xff);
  tgt[4] = static_cast<char>((deflate_size >> 8) & 0xff);
  tgt[5] = static_cast<char>((deflate_size >> 16) & 0xff);

  tgt[6] = static_cast<char>(inflate_size & 0xff);
  tgt[7] = static_cast<char>((inflate_size >> 8) & 0xff);
  tgt[8] = static_cast<char>((inflate_size >> 16) & 0xff);
}

DecompressionEngine::~DecompressionEngine() {}

}

////////////////////////////////////////////////////////////////////////////////

  int CompressionSettings(RCompressionSetting::EAlgorithm::EValues algorithm,
                          int compressionLevel)
  {
    if (compressionLevel < 0) compressionLevel = 0;
    if (compressionLevel > 99) compressionLevel = 99;
    int algo = algorithm;
    if (algorithm >= ROOT::RCompressionSetting::EAlgorithm::kUndefined) algo = 0;
    return algo * 100 + compressionLevel;
  }

  int CompressionSettings(ROOT::ECompressionAlgorithm algorithm,
                          int compressionLevel)
  {
    if (compressionLevel < 0) compressionLevel = 0;
    if (compressionLevel > 99) compressionLevel = 99;
    int algo = algorithm;
    if (algorithm >= ROOT::ECompressionAlgorithm::kUndefinedCompressionAlgorithm) algo = 0;
    return algo * 100 + compressionLevel;
  }
}
