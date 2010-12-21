#ifndef __XRDSYS_PLATFORM_H__
#define __XRDSYS_PLATFORM_H__
/******************************************************************************/
/*                                                                            */
/*                     X r d S y s P l a t f o r m . h h                      */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//        $Id$

// Include stdlib so that ENDIAN macros are defined properly
//
#include <stdlib.h>
#ifdef __linux__
#include <memory.h>
#include <string.h>
#include <sys/types.h>
#include <asm/param.h>
#include <byteswap.h>
#define MAXNAMELEN NAME_MAX
#endif
#ifdef __macos__
#include <AvailabilityMacros.h>
#include <sys/types.h>
#define fdatasync(x) fsync(x)
#define MAXNAMELEN NAME_MAX
#ifndef dirent64
#  define dirent64 dirent
#endif
#ifndef off64_t
#define off64_t int64_t
#endif
#if (!defined(MAC_OS_X_VERSION_10_5) || \
     MAC_OS_X_VERSION_MAX_ALLOWED < MAC_OS_X_VERSION_10_5)
#ifndef stat64
#  define stat64 stat
#endif
#endif
#endif
#ifdef __FreeBSD__
#include <sys/types.h>
#endif

#ifdef __solaris__
#define posix_memalign(memp, algn, sz) \
        ((*memp = memalign(algn, sz)) ? 0 : ENOMEM)
#endif

#if defined(__linux__) || defined(__macos__) || defined(__FreeBSD__)

#define S_IAMB      0x1FF   /* access mode bits */

#define F_DUP2FD F_DUPFD

#define STATFS      statfs
#define STATFS_BUFF struct statfs

#define FS_BLKFACT  4

#define FLOCK_t struct flock

typedef off_t offset_t;

#define GTZ_NULL (struct timezone *)0

#else

#define STATFS      statvfs
#define STATFS_BUFF struct statvfs

#define FS_BLKFACT  1

#define SHMDT_t char *

#define FLOCK_t flock_t

#define GTZ_NULL (void *)0

#endif

#ifdef __linux__

#define SHMDT_t const void *
#endif

// For alternative platforms
//
#ifdef __macos__
#include <AvailabilityMacros.h>
extern char *cuserid(char *s);
#ifndef POLLRDNORM
#define POLLRDNORM  0
#endif
#ifndef POLLRDBAND
#define POLLRDBAND  0
#endif
#ifndef POLLWRNORM
#define POLLWRNORM  0
#endif
#define O_LARGEFILE 0
#define memalign(pgsz,amt) valloc(amt)
#define posix_memalign(memp, algn, sz) \
        ((*memp = memalign(algn, sz)) ? 0 : ENOMEM)
#define SHMDT_t void *
#ifndef EDEADLOCK
#define EDEADLOCK EDEADLK
#endif
#endif

#ifdef __FreeBSD__
#define	O_LARGEFILE 0
typedef off_t off64_t;
#define	memalign(pgsz,amt) valloc(amt)
#endif

// Only sparc platforms have structure alignment problems w/ optimization
// so the h2xxx() variants are used when converting network streams.

#if defined(_BIG_ENDIAN) || defined(__BIG_ENDIAN__) || \
   defined(__IEEE_BIG_ENDIAN) || \
   (defined(__BYTE_ORDER) && __BYTE_ORDER == __BIG_ENDIAN)
#define Xrd_Big_Endian
#ifndef htonll
#define htonll(_x_)  _x_
#endif
#ifndef h2nll
#define h2nll(_x_, _y_) memcpy((void *)&_y_,(const void *)&_x_,sizeof(long long))
#endif
#ifndef ntohll
#define ntohll(_x_)  _x_
#endif
#ifndef n2hll
#define n2hll(_x_, _y_) memcpy((void *)&_y_,(const void *)&_x_,sizeof(long long))
#endif

#elif defined(_LITTLE_ENDIAN) || defined(__LITTLE_ENDIAN__) || \
     defined(__IEEE_LITTLE_ENDIAN) || \
     (defined(__BYTE_ORDER) && __BYTE_ORDER == __LITTLE_ENDIAN)
#if !defined(__GNUC__) || defined(__macos__)

#if !defined(__sun) || (defined(__sun) && !defined(_LP64))
extern "C" unsigned long long Swap_n2hll(unsigned long long x);
#define htonll(_x_) Swap_n2hll(_x_)
#define ntohll(_x_) Swap_n2hll(_x_)
#endif

#else

#ifndef htonll
#define htonll(_x_) __bswap_64(_x_)
#endif
#ifndef ntohll
#define ntohll(_x_) __bswap_64(_x_)
#endif

#endif

#ifndef h2nll
#define h2nll(_x_, _y_) memcpy((void *)&_y_,(const void *)&_x_,sizeof(long long));\
                        _y_ = htonll(_y_)
#endif
#ifndef n2hll
#define n2hll(_x_, _y_) memcpy((void *)&_y_,(const void *)&_x_,sizeof(long long));\
                        _y_ = ntohll(_y_)
#endif

#else
#ifndef WIN32
#error Unable to determine target architecture endianness!
#endif
#endif

#ifndef HAVE_STRLCPY
extern "C"
{extern size_t strlcpy(char *dst, const char *src, size_t size);}
#endif

//
// To make socklen_t portable use SOCKLEN_t
//
#if defined(__solaris__) && !defined(__linux__)
#   if __GNUC__ >= 3 || __GNUC_MINOR__ >= 90
#      define XR__SUNGCC3
#   endif
#endif
#if defined(__linux__)
#   include <features.h>
#   if __GNU_LIBRARY__ == 6
#      ifndef XR__GLIBC
#         define XR__GLIBC
#      endif
#   endif
#endif
#if defined(__MACH__) && defined(__i386__)
#   define R__GLIBC
#endif
#if defined(_AIX) || \
   (defined(XR__SUNGCC3) && !defined(__arch64__))
#   define SOCKLEN_t size_t
#elif defined(XR__GLIBC) || \
   defined(__FreeBSD__) || \
   (defined(XR__SUNGCC3) && defined(__arch64__)) || defined(__macos__)
#   ifndef SOCKLEN_t
#      define SOCKLEN_t socklen_t
#   endif
#elif !defined(SOCKLEN_t)
#   define SOCKLEN_t int
#endif

#ifdef _LP64
#define PTR2INT(x) static_cast<int>((long long)x)
#else
#define PTR2INT(x) int(x)
#endif

#ifdef WIN32
#include "XrdSys/XrdWin32.hh"
#define Netdata_t void *
#define Sokdata_t char *
#define IOV_INIT(data,dlen) dlen,data
#define MAKEDIR(path,mode) mkdir(path)
#define net_errno WSAGetLastError()
#else
#define O_BINARY 0
#define Netdata_t char *
#define Sokdata_t void *
#define IOV_INIT(data,dlen) data,dlen
#define MAKEDIR(path,mode) mkdir(path,mode)
#define net_errno errno
#endif

#ifdef WIN32
#define MAXNAMELEN 256
#define MAXPATHLEN 1024
#else
#include <sys/param.h>
#endif
// The following gets arround a relative new gcc compiler bug
//
#define XRDABS(x) (x < 0 ? -x : x)

#endif  // __XRDSYS_PLATFORM_H__
