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

#if defined(__cplusplus)
namespace ROOT {
#endif

/// The global settings depend on a global variable named R__ZipMode which can be
/// modified by a global function named R__SetZipMode. Both are defined in Bits.h.
///
///  - The default is to use the global setting and the default of the global
///    setting is to use the ZLIB compression algorithm.
///  - The LZMA algorithm (from the XZ package) is also available. The LZMA
///    compression usually results in greater compression factors, but takes
///    more CPU time and memory when compressing. LZMA memory usage is particularly
///    high for compression levels 8 and 9.
///  - Finally, the LZ4 package results in worse compression ratios
///    than ZLIB but achieves much faster decompression rates.
///
/// The current algorithms support level 1 to 9. The higher the level the greater
/// the compression and more CPU time and memory resources used during compression.
/// Level 0 means no compression.
///
/// Recommendation for the compression algorithm's levels:
///  - ZLIB is recommended to be used with compression level 1 [101] 
///  - LZMA is recommended to be used with compression level 7-8 (higher is better, 
///   since in the case of LZMA we don't care about compression/decompression speed)
///   [207 - 208]
///  - LZ4 is recommended to be used with compression level 4 [404]


enum ECompressionAlgorithm {
   /// Use the global compression setting
   kUseGlobalCompressionSetting,
   /// Use ZLIB compression
   kZLIB,
   /// Use LZMA compression
   kLZMA,
   /// Use the old compression algorithm
   kOldCompressionAlgo,
   /// Use LZ4 compression
   kLZ4,
   /// Undefined compression algorithm (must be kept the last of the list in case a new algorithm is added).
   kUndefinedCompressionAlgorithm
};

/// Deprecated name, do *not* use:
static const enum ECompressionAlgorithm kUseGlobalSetting = kUseGlobalCompressionSetting;

#if defined(__cplusplus)

   int CompressionSettings(ECompressionAlgorithm algorithm,
                           int compressionLevel);
}
#endif

#endif
