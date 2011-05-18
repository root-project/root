/* @(#)root/clib:$Id$ */
/* Author: Fons Rademakers  10/12/2000 */

/*
   Write formatted text to buffer 'string', using format string 'format'.
   Returns number of characters written, or -1 if truncated.
   Format string is understood as defined in ANSI C.
*/

#ifndef CINT_snprintf
#define CINT_snprintf

#if defined(__MSC_VER)
#define CINT_NEED_SNPRINTF
#   if _MSC_VER >= 1400
#     define CINT_DONTNEED_VSNPRINTF
#   endif
#elif defined(__hpux)
#   ifdef R__HPUX10
#      define CINT_NEED_SNPRINTF
#   endif
#elif defined(__alpha) && !defined(linux)
#   ifndef R__TRUE64
#      define CINT_NEED_SNPRINTF
#   endif
#elif defined(__Lynx__) && defined(__powerpc__)
#   define CINT_NEED_SNPRINTF
#elif defined(_HIUX_SOURCE)
#   define CINT_NEED_SNPRINTF
#elif defined(__SC__)
#   if defined(WIN32)
#      define CINT_NEED_SNPRINTF
#   endif
#endif

#ifndef CINT_NEED_SNPRINTF
#include <stdio.h>
#define G__snprintf snprintf
#define G__vsnprintf vsprintf
#else

#include <stdio.h>
#include <stdarg.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef CINT_DONTNEED_VSNPRINTF
int vsnprintf(char *string, size_t length, const char *format, va_list args);
#else
#define G__vsnprintf vsprintf
#endif
int G__snprintf(char *string, size_t length, const char *format, ...);

#ifdef __cplusplus
}
#endif

#endif /* CINT_NEED_SNPRINTF */

#endif /* CINT_snprintf */
