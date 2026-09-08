/* @(#)root/clib:$Id$ */
/* Author: Fons Rademakers  10/12/2000 */

/*
   Write formatted text to buffer 'string', using format string 'format'.
   Returns number of characters written, or -1 if truncated.
   Format string is understood as defined in ANSI C.
*/

#ifndef ROOT_snprintf
#define ROOT_snprintf

#warning "This header is deprecated and will be removed in ROOT 6.44, use instead <cstdio>"

#include <ROOT/RConfig.hxx>
#include <stdio.h>
#ifdef NEED_SNPRINTF
#error "ROOT no longer provides fallback implementation for snprintf. NEED_SNPRINTF should not be defined."
#endif /* NEED_SNPRINTF */

#endif /* ROOT_snprintf */
