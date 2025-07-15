/* @(#)root/foundation:$Id$ */

/*************************************************************************
 * Copyright (C) 1995-2014, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RtypesCore
#define ROOT_RtypesCore

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// RtypesCore                                                           //
//                                                                      //
// Basic types used by ROOT and required by TInterpreter.               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

/**
 * \file
 * \brief Basic types used by ROOT and required by TInterpreter.
 * It ensures a portable fixed data type size across systems, since in
 * the early days, sizeof(int) could be 2 or 4 depending on the architecture.
 * \deprecated For future designs, unless for very specific needs, consider using instead standard
 * fixed-width classes from <cstdint> such as `std::int16_t`, `std::int32_t` or from <cstdfloat> for floating types.
 * \warning `Long_t` has not the same width across platforms, so it should be avoided if portability is envisioned.
 * Also derived classes such as `TArrayL`. Use instead `Long64_t` or `TArrayL64`, or `std::int64_t`.
 * Likewise with `ULong_t`.
 * \warning In some architectures, `std::int64_t` may have a different underlying data type (long vs int) than in others
 * and may lead to a different StreamerInfo than in others, thus it might be convenient to use (U)Long64_t instead.
 * Hence, full fledge embrace of the std::types is actually waiting on ROOT I/O to be extended to support them explicitly.
 */

#include <ROOT/RConfig.hxx>

#include "DllImport.h"

#ifndef R__LESS_INCLUDES
#include <cstddef> // size_t, NULL
#endif

//---- Tag used by rootcling to determine constructor used for I/O.

class TRootIOCtor;

//---- types -------------------------------------------------------------------

typedef char           Char_t;      ///< Signed Character 1 byte (char) \deprecated Consider replacing with `char` or `std::int8_t`
typedef unsigned char  UChar_t;     ///< Unsigned Character 1 byte (unsigned char) \deprecated Consider replacing with `unsigned char` or `std::uint8_t`
typedef short          Short_t;     ///< Signed Short integer 2 bytes (short) \deprecated Consider replacing with `short` or `std::int16_t`
typedef unsigned short UShort_t;    ///< Unsigned Short integer 2 bytes (unsigned short) \deprecated Consider replacing with `unsigned short` or `std::uint16_t`
#ifdef R__INT16
typedef long           Int_t;       ///< Signed integer 4 bytes \deprecated Consider replacing with `std::int32_t`
typedef unsigned long  UInt_t;      ///< Unsigned integer 4 bytes \deprecated Consider replacing with `std::int32_t`
#else
typedef int            Int_t;       ///< Signed integer 4 bytes (int) \deprecated Consider replacing with `std::int32_t`
typedef unsigned int   UInt_t;      ///< Unsigned integer 4 bytes (unsigned int) \deprecated Consider replacing with `std::uint32_t`
#endif
#ifdef R__B64    // Note: Long_t and ULong_t are currently not portable types
typedef int            Seek_t;      ///< File pointer (int).
typedef long           Long_t;      ///< Signed long integer 8 bytes (long). Size depends on architecture \deprecated Consider replacing with `long`
typedef unsigned long  ULong_t;     ///< Unsigned long integer 8 bytes (unsigned long). Size depends on architecture \deprecated Consider replacing with `unsigned long`
#else
typedef int            Seek_t;      ///< File pointer (int). 
typedef long           Long_t;      ///< Signed long integer 4 bytes (long). Size depends on architecture \deprecated Consider replacing with `long`
typedef unsigned long  ULong_t;     ///< Unsigned long integer 4 bytes (unsigned long). Size depends on architecture \deprecated Consider replacing with `unsigned long`
#endif
typedef float          Float_t;     ///< Float 4 bytes (float) \deprecated Consider replacing with `float`.
typedef float          Float16_t;   ///< Float 4 bytes in memory, written to disk as 3 bytes (24-bits) by default or as a 4 bytes fixed-point-arithmetic Int_t (32-bits) if range was customized, with a truncated mantissa (12-bit by default in memory), and (7+1)-bits of exponent \warning Do not confuse Float16_t on file representation with a half-float such as std::float16_t.
typedef double         Double_t;    ///< Double 8 bytes \deprecated Consider replacing with `double`.
typedef double         Double32_t;  ///< Double 8 bytes in memory, written to disk as a 4 bytes Float_t (32-bits) by default, or as 3 bytes (24-bits) float if range is customized, with a truncated mantissa (24-bit by default in memory, less if range is customized), and (7+1)-bits of exponent \warning Do not confuse Double32_t on file representation with a single-precision float such as std::float32_t
typedef long double    LongDouble_t;///< Long Double (not portable) \deprecated Consider replacing with `long double`.
typedef char           Text_t;      ///< General string (char)
typedef bool           Bool_t;      ///< Boolean (0=false, 1=true) (bool) \deprecated Consider replacing with `bool`.
typedef unsigned char  Byte_t;      ///< Byte (8 bits) (unsigned char) \deprecated Consider replacing with `unsigned char` or `std::byte`.
typedef short          Version_t;   ///< Class version identifier (short)
typedef const char     Option_t;    ///< Option string (const char)
typedef int            Ssiz_t;      ///< String size (currently int)
typedef float          Real_t;      ///< TVector and TMatrix element type (float) \deprecated Consider replacing with `float`.
typedef long long           Long64_t;///< Portable signed long integer 8 bytes \deprecated Consider replacing with `long long` or `std::int64_t` (unless you are worried about different StreamerInfos in different platforms).
typedef unsigned long long ULong64_t;///< Portable unsigned long integer 8 bytes \deprecated Consider replacing with `unsigned long long` or `std::uint64_t` (unless you are worried about different StreamerInfos in different platforms).
#ifdef _WIN64
typedef long long      Longptr_t;     ///< Integer large enough to hold a pointer (platform-dependent)
typedef unsigned long long ULongptr_t;///< Unsigned integer large enough to hold a pointer (platform-dependent)
#else
typedef long           Longptr_t;   ///< Integer large enough to hold a pointer (platform-dependent)
typedef unsigned long  ULongptr_t;  ///< Unsigned integer large enough to hold a pointer (platform-dependent)
#endif
typedef double         Axis_t;      ///< Axis values type (double)
typedef double         Stat_t;      ///< Statistics type (double)

typedef short          Font_t;      ///< Font number (short)
typedef short          Style_t;     ///< Style number (short)
typedef short          Marker_t;    ///< Marker number (short)
typedef short          Width_t;     ///< Line width (short)
typedef short          Color_t;     ///< Color number (short)
typedef short          SCoord_t;    ///< Screen coordinates (short)
typedef double         Coord_t;     ///< Pad world coordinates (double)
typedef float          Angle_t;     ///< Graphics angle (float)
typedef float          Size_t;      ///< Attribute size (float)

//---- constants ---------------------------------------------------------------

constexpr Bool_t kTRUE = true;  ///< \deprecated Consider replacing with `true`
constexpr Bool_t kFALSE = false;///< \deprecated Consider replacing with `false`

constexpr Int_t kMaxUChar = UChar_t(~0);  ///< \deprecated Consider replacing with `std::numeric_limits<unsigned char>::max()` (or `std::uint8_t`)
constexpr Int_t kMaxChar = kMaxUChar >> 1;///< \deprecated Consider replacing with `std::numeric_limits<char>::max()` (or `std::int8_t`)
constexpr Int_t kMinChar = -kMaxChar - 1; ///< \deprecated Consider replacing with `std::numeric_limits<char>::lowest()` (or `std::int8_t`)

constexpr Int_t kMaxUShort = UShort_t(~0);  ///< \deprecated Consider replacing with `std::numeric_limits<unsigned short>::max()` (or `std::uint16_t`)
constexpr Int_t kMaxShort = kMaxUShort >> 1;///< \deprecated Consider replacing with `std::numeric_limits<short>::max()` (or `std::int16_t`)
constexpr Int_t kMinShort = -kMaxShort - 1; ///< \deprecated Consider replacing with `std::numeric_limits<short>::lowest()` (or `std::int16_t`)

constexpr UInt_t kMaxUInt = UInt_t(~0);        ///< \deprecated Consider replacing with `std::numeric_limits<unsigned int>::max()` (or `std::uint32_t`)
constexpr Int_t kMaxInt = Int_t(kMaxUInt >> 1);///< \deprecated Consider replacing with `std::numeric_limits<int>::max()` (or `std::int32_t`)
constexpr Int_t kMinInt = -kMaxInt - 1;        ///< \deprecated Consider replacing with `std::numeric_limits<int>::lowest()` (or `std::int32_t`)

constexpr ULong_t kMaxULong = ULong_t(~0);         ///< \deprecated Consider replacing with `std::numeric_limits<unsigned long>::max()`
constexpr Long_t kMaxLong = Long_t(kMaxULong >> 1);///< \deprecated Consider replacing with `std::numeric_limits<long>::max()`
constexpr Long_t kMinLong = -kMaxLong - 1;         ///< \deprecated Consider replacing with `std::numeric_limits<long>::lowest()`

constexpr ULong64_t kMaxULong64 = ULong64_t(~0LL);         ///< \deprecated Consider replacing with `std::numeric_limits<unsigned long long>::max()` (or `std::uint64_t`)
constexpr Long64_t kMaxLong64 = Long64_t(kMaxULong64 >> 1);///< \deprecated Consider replacing with `std::numeric_limits<long long>::max()` (or `std::int64_t`)
constexpr Long64_t kMinLong64 = -kMaxLong64 - 1;           ///< \deprecated Consider replacing with `std::numeric_limits<long long>::lowest()` (or `std::int64_t`)

constexpr ULong_t kBitsPerByte = 8; ///< \deprecated Consider replacing with `std::numeric_limits<unsigned char>::digits`.
constexpr Ssiz_t kNPOS = ~(Ssiz_t)0;///< The equivalent of `std::string::npos` for the ROOT class TString. \note Consider using std::string instead of TString whenever possible

//---- debug global ------------------------------------------------------------

R__EXTERN Int_t gDebug;///< Global variable setting the debug level. Set to 0 to disable, increase it in steps of 1 to increase the level of debugging-printing verbosity when running ROOT commands


#endif
