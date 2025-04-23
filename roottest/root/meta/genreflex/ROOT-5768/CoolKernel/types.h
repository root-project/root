// $Id: types.h,v 1.54 2009-12-17 18:50:42 avalassi Exp $
#ifndef COOLKERNEL_TYPES_H
#define COOLKERNEL_TYPES_H

// Include files
#include <climits> // For min/max values (e.g. SCHAR_MIN) on gcc43
#include <string>
#include "CoralBase/Blob.h"

namespace cool {

  /** @file types.h
   *
   * Type definitions of the basic C++ types that can be stored in COOL.
   * Also includes the definitions of types that are not yet supported.
   *
   * @author Andrea Valassi, Marco Clemencic and Sven A. Schmidt
   * @date 2004-11-05
   */

  // -------------------------------------------------------------------------
  /// Type definitions for all basic C++ types known by COOL (both those
  /// already supported and those candidate to be supported in the future).
  // -------------------------------------------------------------------------

  typedef bool Bool;

  typedef signed char SChar;
  typedef unsigned char UChar;

  typedef short Int16;
  typedef unsigned short UInt16;

  typedef int Int32;
  typedef unsigned int UInt32;

  typedef long long Int64;
  typedef unsigned long long UInt64;

  typedef UInt64 UInt63;

  typedef float Float;
  typedef double Double;

  typedef std::string String255;
  typedef std::string String4k;
  typedef std::string String64k;
  typedef std::string String16M;

  typedef coral::Blob Blob64k;
  typedef coral::Blob Blob16M;

  // -------------------------------------------------------------------------
  /// Constant definitions for the min/max values of a few supported types.
  /// [NB The cool::SChar type is defined as 'signed char' instead of 'char'.
  /// The C standard allows 'char' to be either signed or unsigned.
  /// The corresponding min/max values depend on __CHAR_UNSIGNED__,
  /// see for instance /usr/include/limits.h on Linux.]
  // -------------------------------------------------------------------------

  const SChar SCharMin = SCHAR_MIN; // -127
  const SChar SCharMax = SCHAR_MAX; // +128
  const UChar UCharMin = 0;
  const UChar UCharMax = UCHAR_MAX; // +255

  const Int16 Int16Min  = SHRT_MIN; // -32768
  const Int16 Int16Max  = SHRT_MAX; // +32767
  const UInt16 UInt16Min = 0;
  const UInt16 UInt16Max = USHRT_MAX; // +65535

  const Int32 Int32Min  = INT_MIN; // -2147483648
  const Int32 Int32Max  = INT_MAX; // +2147483647
  const UInt32 UInt32Min = 0;
  const UInt32 UInt32Max = UINT_MAX; // +4294967295

#ifdef WIN32
  // Supported platforms: win32_vc71
  // Marco: gccxml does not like the Microsoft constants.
  //const Int64  Int64Min  = _I64_MIN;                 // -9223372036854775808
  //const Int64  Int64Max  = _I64_MAX;                 // +9223372036854775807
  //const UInt64 UInt64Min = 0;
  //const UInt64 UInt64Max = _UI64_MAX;                // +18446744073709551615
  const Int64 Int64Min  = 0x8000000000000000ll; // -9223372036854775808
  const Int64 Int64Max  = 0x7fffffffffffffffll; // +9223372036854775807
  const UInt64 UInt64Min = 0;
  const UInt64 UInt64Max = 0xffffffffffffffffll; // +18446744073709551615
#else
#  if defined LONG_LONG_MAX
  // Supported platforms: slc3_ia32, slc4_ia32, slc4_amd64
  const Int64 Int64Min  = LONG_LONG_MIN; // -9223372036854775808
  const Int64 Int64Max  = LONG_LONG_MAX; // +9223372036854775807
  const UInt64 UInt64Min = 0;
  const UInt64 UInt64Max = ULONG_LONG_MAX; // +18446744073709551615
#  else
  // Supported platforms: osx104_ppc (why is LONG_LONG_MAX not defined?...)
  // See /usr/lib/gcc/powerpc-apple-darwin8/4.0.1/include/limits.h
  const Int64 Int64Min  = -__LONG_LONG_MAX__-1LL; // -9223372036854775808
  const Int64 Int64Max  = __LONG_LONG_MAX__; // +9223372036854775807
  const UInt64 UInt64Min = 0;
  const UInt64 UInt64Max = __LONG_LONG_MAX__*2ULL+1ULL; // +18446744073709551615
#  endif
#endif

  const UInt63 UInt63Min = UInt64Min;
  const UInt63 UInt63Max = Int64Max;

  // -------------------------------------------------------------------------

}
#endif
