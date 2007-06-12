// @(#)root/proofd:$Name:  $:$Id: $
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <sys/stat.h>
#include <pwd.h>
#include <sys/resource.h>
#include <sys/file.h>
#include <dirent.h>

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
#   define HASNOT_INETATON
#   define INADDR_NONE (UInt_t)-1
#endif

#include <dlfcn.h>
#if !defined(__APPLE__)
#include <link.h>
#endif

#if defined(linux) || defined(__sun) || defined(__sgi) || \
    defined(_AIX) || defined(__FreeBSD__) || defined(__OpenBSD__) || \
    defined(__APPLE__) || defined(__MACH__) || defined(cygwingcc)
#include <grp.h>
#include <sys/types.h>
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
static const char *gLIBPATH = "SHLIB_PATH";
#elif defined(_AIX)
static const char *gLIBPATH = "LIBPATH";
#elif defined(__APPLE__)
static const char *gLIBPATH = "DYLD_LIBRARY_PATH";
#else
static const char *gLIBPATH = "LD_LIBRARY_PATH";
#endif

#endif
