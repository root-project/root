/* @(#)root/clib:$Name$:$Id$ */

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef MMCONFIG_H
#define MMCONFIG_H

#ifndef ROOT_RConfig
#include "RConfig.h"
#endif

#ifndef WIN32
#  ifndef INVALID_HANDLE_VALUE
#    define INVALID_HANDLE_VALUE -1
#  endif
#endif

#if defined (R__MAC)
#   define HAVE_STDDEF_H
#   define HAVE_LIMITS_H
#   define NO_SBRK_MALLOC
#elif defined(R__UNIX)
#   define HAVE_UNISTD_H
#   define HAVE_STDLIB_H
#   define HAVE_STDDEF_H
#   define HAVE_LIMITS_H
#   define HAVE_MMAP
#   define NO_SBRK_MALLOC
#elif defined (R__VMS)
#   define HAVE_UNISTD_H
#   define HAVE_STDLIB_H
#   define HAVE_STDDEF_H
#   define NO_SBRK_MALLOC
#else
#   define HAVE_STDDEF_H
#   define HAVE_LIMITS_H
#   define HAVE_MMAP
#   define NO_SBRK_MALLOC
#endif

#endif
