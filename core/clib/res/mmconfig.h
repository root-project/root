/* @(#)root/clib:$Id$ */

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef MMCONFIG_H
#define MMCONFIG_H

#include <ROOT/RConfig.h>

#ifndef WIN32
#  ifndef INVALID_HANDLE_VALUE
#    define INVALID_HANDLE_VALUE -1
#  endif
#endif

#if defined(R__UNIX)
#   define R__HAVE_UNISTD_H
#   define R__HAVE_STDLIB_H
#   define R__HAVE_STDDEF_H
#   define R__HAVE_LIMITS_H
#   define R__HAVE_MMAP
#   define NO_SBRK_MALLOC
#else
#   define R__HAVE_STDDEF_H
#   define R__HAVE_LIMITS_H
#   define R__HAVE_MMAP
#   define NO_SBRK_MALLOC
#endif

#endif
