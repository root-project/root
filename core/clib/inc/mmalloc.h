/* @(#)root/clib:$Id$ */
/* Author: */

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef MMALLOC_H
#define MMALLOC_H 1

/*  FIXME:  If <stddef.h> doesn't exist, you'll need to do something
            to define size_t before including this file.  Like upgrading
            to a system with an ANSI C environment. */

#include "mmconfig.h"

#ifdef WIN32
#  include <windows.h>
#endif

#ifdef R__HAVE_STDDEF_H
#  include <stddef.h>
#endif

#define PTR                 void *
#define PARAMS(paramlist)   paramlist

#ifdef WIN32
   extern struct mstats mmstats PARAMS ((PTR));
#endif

#ifdef  __cplusplus
extern "C" {
#endif

/* Allocate SIZE bytes of memory.  */

extern PTR mmalloc PARAMS ((PTR, size_t));

/* Re-allocate the previously allocated block in PTR, making the new block
   SIZE bytes long.  */

extern PTR mrealloc PARAMS ((PTR, PTR, size_t));

/* Allocate NMEMB elements of SIZE bytes each, all initialized to 0.  */

extern PTR mcalloc PARAMS ((PTR, size_t, size_t));

/* Free a block allocated by `mmalloc', `mrealloc' or `mcalloc'.  */

extern void mfree PARAMS ((PTR, PTR));

/* Allocate SIZE bytes allocated to ALIGNMENT bytes.  */

extern PTR mmemalign PARAMS ((PTR, size_t, size_t));

/* Allocate SIZE bytes on a page boundary.  */

extern PTR mvalloc PARAMS ((PTR, size_t));

/* Activate a standard collection of debugging hooks.  */

extern int mmcheck PARAMS ((PTR, void (*) (void)));

/* Pick up the current statistics is no longer supported via mmstats. (see FIXME elsewhere) */
/* This cause a problem with VC++ (warning C4190) if extern "C" and with clang:
include/mmalloc.h:73:25: warning: 'mmstats' has C-linkage specified, but returns user-defined type 'struct mstats' which is incompatible with C [-Wreturn-type-c-linkage]
   This is because mstats is only (intentionally see the FIXME) declared in mmprivate.h
   and C (or just VC++ and clang in extern "C" mode) requires the struct to be fully 
   declared before it is used (likely so that it can know whether it is a POD or not).
   Since mmprivate.h first include mmalloc.h, even including expliclty mmprivate.h
   we still can not get in a position where struct mstats is fully declared before
   the next line is parsed ... Since the FIXME in mmprivate.h says that anyhow
   the data should not be provided using this struct mstats ... we just comment
   this comment it out.
*/
   // extern struct mstats mmstats PARAMS ((PTR));

#ifndef WIN32
   extern PTR mmalloc_attach PARAMS ((int, PTR, int));
#else
   extern PTR mmalloc_attach PARAMS ((HANDLE, PTR, int));
#endif

extern PTR mmalloc_detach PARAMS ((PTR));

extern int mmalloc_update_mapping PARAMS ((PTR));

extern int mmalloc_setkey PARAMS ((PTR, int, PTR));

extern PTR mmalloc_getkey PARAMS ((PTR, int));

extern int mmtrace PARAMS ((void));

#ifdef  __cplusplus
}
#endif

#endif  /* MMALLOC_H */
