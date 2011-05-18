#ifndef __XPTYPES_H
#define __XPTYPES_H

//        $Id$

// Full range type compatibility work done by Gerardo Ganis, CERN.

// Typical data types
//
// Only char and short are truly portable types
typedef unsigned char  kXR_char;
typedef short          kXR_int16;
typedef unsigned short kXR_unt16;

// Signed integer 4 bytes
//
#ifndef XR__INT16
#   if defined(LP32) || defined(__LP32) || defined(__LP32__) || \
       defined(BORLAND)
#      define XR__INT16
#   endif
#endif
#ifndef XR__INT64
#   if defined(ILP64) || defined(__ILP64) || defined(__ILP64__)
#      define XR__INT64
#   endif
#endif
#if defined(XR__INT16)
typedef long           kXR_int32;
typedef unsigned long  kXR_unt32;
#elif defined(XR__INT64)
typedef int32          kXR_int32;
typedef unsigned int32 kXR_unt32;
#else
typedef int            kXR_int32;
typedef unsigned int   kXR_unt32;
#endif

// Signed integer 8 bytes
//
//#if defined(_WIN32)
//typedef __int64        kXR_int64;
//#else
typedef long long      kXR_int64;
//#endif
#endif
