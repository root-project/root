// @(#)root/zip:$Id$
// Author: David Dagenhart   May 2011

/*************************************************************************
 * Copyright (C) 1995-2011, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_Compression
#define ROOT_Compression

namespace ROOT {

   // The global settings depend on a global variable named
   // R__ZipMode which can be modified by a global function
   // named R__SetZipMode.  Both are defined in Bits.h.
   // The default is to use the global setting
   // and the default of the global setting is to use the
   // ZLIB compression algorithm.  The LZMA algorithm
   // (from the XZ package) is also available.
   // The LZMA compression usually results
   // in greater compression factors, but takes more CPU time
   // and memory when compressing.  LZMA memory usage is particularly
   // high for compression levels 8 and 9.
   //
   // Finally, the LZ4 package results in worse compression ratios
   // than ZLIB but achieves much faster decompression rates.
   //
   // The current algorithms support level 1 to 9. The higher
   // the level the greater the compression and more CPU time
   // and memory resources used during compression. Level 0
   // means no compression.
   enum ECompressionAlgorithm { kUseGlobalSetting,
                                kZLIB,
                                kLZMA,
                                kOldCompressionAlgo,
                                kLZ4,
                                // if adding new algorithm types,
                                // keep this enum value last
                                kUndefinedCompressionAlgorithm
   };

   int CompressionSettings(ECompressionAlgorithm algorithm,
                           int compressionLevel);
}

#endif
