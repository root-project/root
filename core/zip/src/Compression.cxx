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

namespace ROOT {

////////////////////////////////////////////////////////////////////////////////

  int CompressionSettings(RCompressionSetting::EAlgorithm::EValues algorithm, int compressionLevel)
  {
    if (compressionLevel < 0) compressionLevel = 0;
    if (compressionLevel > 99) compressionLevel = 99;
    int algo = algorithm;
    if (algorithm >= ROOT::RCompressionSetting::EAlgorithm::kUndefined) algo = 0;
    return algo * 100 + compressionLevel;
  }

  std::string RCompressionSetting::AlgorithmToString(RCompressionSetting::EAlgorithm::EValues algorithm)
  {
     switch (algorithm) {
     case EAlgorithm::EValues::kZLIB: return "zlib";
     case EAlgorithm::EValues::kLZMA: return "LZMA";
     case EAlgorithm::EValues::kOldCompressionAlgo: return "Old compression algorithm";
     case EAlgorithm::EValues::kLZ4: return "lz4";
     case EAlgorithm::EValues::kZSTD: return "zstd";
     default: return "Undefined compression algorithm";
     }
  }
}
