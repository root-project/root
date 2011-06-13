#include "Compression.h"

namespace ROOT {

//______________________________________________________________________________
  int CompressionSettings(ECompressionAlgorithm algorithm,
                          int compressionLevel)
  {
    if (compressionLevel < 0) compressionLevel = 0;
    if (compressionLevel > 99) compressionLevel = 99;
    int algo = algorithm;
    if (algorithm >= ROOT::kUndefinedCompressionAlgorithm) algo = 0;
    return algo * 100 + compressionLevel;
  }
}
