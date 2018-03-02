// Author: Guilherme Amadio, Enrico Guiraud, Danilo Piparo CERN  2/2018

/*************************************************************************
 * Copyright (C) 1995-2016, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TSNAPSHOTOPTIONS
#define ROOT_TSNAPSHOTOPTIONS

#include <Compression.h>
#include <ROOT/RStringView.hxx>
#include <string>

namespace ROOT {
namespace Experimental {
namespace TDF {
/// A collection of options to steer the creation of the dataset on file
struct TSnapshotOptions {
   using ECAlgo = ::ROOT::ECompressionAlgorithm;
   TSnapshotOptions() = default;
   TSnapshotOptions(const TSnapshotOptions &) = default;
   TSnapshotOptions(TSnapshotOptions &&) = default;
   TSnapshotOptions(std::string_view mode, ECAlgo comprAlgo, int comprLevel, int autoFlush, int splitLevel)
      : fMode(mode), fCompressionAlgorithm(comprAlgo), fCompressionLevel{comprLevel}, fAutoFlush(autoFlush),
        fSplitLevel(splitLevel)
   {
   }
   std::string fMode = "RECREATE";             //< Mode of creation of output file
   ECAlgo fCompressionAlgorithm = ROOT::kZLIB; //< Compression algorithm of output file
   int fCompressionLevel = 1;                  //< Compression level of output file
   int fAutoFlush = 0;                         //< AutoFlush value for output tree
   int fSplitLevel = 99;                       //< Split level of output tree
};
}
}
}

#endif
