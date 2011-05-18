// @(#)root/editline:$Id$
// Author: Mary-Louise Gill, 2009

/*************************************************************************
 * Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#define HAVE_GETLINE 1
#define HAVE_FLOCKFILE 1

#ifdef __sun
# define HAVE_SYS_CDEFS_H 0
#endif
#ifdef __CYGWIN__
#include <sys/ioctl.h>
#endif

#if defined(__sun) || defined(__CYGWIN__)
extern "C" {
   typedef void (*sig_t)(int);
}
#endif


////////////////////////////////////////////////////////////////////////
// most of this #ifdef code is to work around my own preference of
// #if FOO, instead of #ifdef FOO, in client code. Since toc
// does a define to 0 on false (which i think is sane), we need
// to UNSET those vars which toc sets to zero, to accomodate
// this source tree.
#ifndef HAVE_SYS_TYPES_H
# define HAVE_SYS_TYPES_H 1
#endif
#if (0 == HAVE_SYS_TYPES_H)
# undef HAVE_SYS_TYPES_H
#endif

#ifndef HAVE_SYS_CDEFS_H
# define HAVE_SYS_CDEFS_H 1
#endif
#if (0 == HAVE_SYS_CDEFS_H)
# undef HAVE_SYS_CDEFS_H
#endif

