/* @(#)root/clib:$Id$ */
/* Author: Fons Rademakers  20/9/2010 */

/*
   Inlcude file for strlcpy and strlcat. They are in string.h on systems
   that have these function (BSD based systems).
*/

#ifndef ROOT_strlcpy
#define ROOT_strlcpy

#include <ROOT/RConfig.hxx>

#ifndef HAS_STRLCPY

#ifndef WIN32
#   include <unistd.h>
#else
#   include <sys/types.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

size_t strlcpy(char *dst, const char *src, size_t siz);
size_t strlcat(char *dst, const char *src, size_t siz);

#ifdef __cplusplus
}
#endif
#endif /* HAS_STRLCPY */

#endif /* ROOT_strlcpy */
