// @(#)root/proofd:$Id$
// Author: G. Ganis  June 2007

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_XrdProofdPlatform
#define ROOT_XrdProofdPlatform

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdProofdPlatform                                                    //
//                                                                      //
// Authors: G. Ganis, CERN, 2007                                        //
//                                                                      //
// System settings used in XrdProofd classes, possibly platform         //
// dependent                                                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_RConfig
#include "RConfig.h"
#endif

// 32 or 64 bits
#if ((defined(__hpux) && defined(__LP64__)) || \
     (defined(linux) && (defined(__ia64__) || defined(__x86_64__))) || \
     (defined(linux) && defined(__powerpc__) && defined(R__ppc64)) || \
     (defined(__APPLE__) && (defined(__ppc64__) || defined(__x86_64__))))
#  define XPD__B64
#endif

#ifdef __APPLE__
#   ifndef __macos__
#      define __macos__
#   endif
#endif
#ifdef __sun
#   ifndef __solaris__
#      define __solaris__
#   endif
#endif

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#ifndef ROOT_XrdFour
#  include <sys/socket.h>
#  include <netinet/in.h>
#endif
#include <sys/stat.h>
#include <sys/un.h>
#include <pwd.h>
#include <sys/resource.h>
#include <sys/file.h>
#include <dirent.h>
#include <libgen.h>

// Bypass Solaris ELF madness
//
#if (defined(SUNCC) || defined(__sun))
#include <sys/isa_defs.h>
#if defined(_ILP32) && (_FILE_OFFSET_BITS != 32)
#undef  _FILE_OFFSET_BITS
#define _FILE_OFFSET_BITS 32
#undef  _LARGEFILE_SOURCE
#endif
#endif

// System info on Solaris
#if (defined(SUNCC) || defined(__sun)) && !defined(__KCC)
#   define XPD__SUNCC
#   include <sys/systeminfo.h>
#   include <sys/filio.h>
#   include <sys/sockio.h>
#   include <fcntl.h>
#   define HASNOT_INETATON
#   ifndef INADDR_NONE
#   define INADDR_NONE (UInt_t)-1
#   endif
#endif

#include <dlfcn.h>
#if !defined(__APPLE__)
#include <link.h>
#endif

#if defined(linux) || defined(__sun) || defined(__sgi) || \
    defined(_AIX) || defined(__FreeBSD__) || defined(__OpenBSD__) || \
    defined(__APPLE__) || defined(__MACH__) || defined(cygwingcc)
#include <grp.h>
#endif

// For process info
#if defined(__sun)
#include <procfs.h>
#elif defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__APPLE__)
#include <sys/sysctl.h>
#endif

// Poll
#include <sys/poll.h>

// Name of the env variable used to define the library path
#if defined(__hpux) || defined(_HIUX_SOURCE)
#define XPD_LIBPATH "SHLIB_PATH"
#elif defined(_AIX)
#define XPD_LIBPATH "LIBPATH"
#elif defined(__APPLE__)
#define XPD_LIBPATH "DYLD_LIBRARY_PATH"
#else
#define XPD_LIBPATH "LD_LIBRARY_PATH"
#endif

// Time related
#include <sys/time.h>
#include <utime.h>

// Macros to check ranges
#ifdef XPD__B64
#  define XPD_LONGOK(x) (1)
#else
#  define XPD_LONGOK(x) (x > LONG_MIN && x < LONG_MAX)
#endif

// Wait related
#include <sys/wait.h>

#endif
