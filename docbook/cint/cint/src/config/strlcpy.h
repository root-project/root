/* @(#)root/clib:$Id$ */
/* Author: Fons Rademakers  20/9/2010 */

/*
   Inlcude file for strlcpy and strlcat. They are in string.h on systems
   that have these function (BSD based systems).
*/

#ifndef CINT_strlcpy
#define CINT_strlcpy

#if defined(__FreeBSD__) ||  defined(__OpenBSD__) || defined(__APPLE__)
#define CINT_HAS_STRLCPY
#endif

#include <string.h>

#ifdef CINT_HAS_STRLCPY

#define G__strlcpy strlcpy
#define G__strlcat strlcat

#else

#ifndef WIN32
#   include <unistd.h>
#else
#   include <sys/types.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

size_t G__strlcpy(char *dst, const char *src, size_t siz);
size_t G__strlcat(char *dst, const char *src, size_t siz);

#ifdef __cplusplus
}
#endif

#endif /* CINT_HAS_STRLCPY */

#endif /* CINT_strlcpy */
