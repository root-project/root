/* config.h.in.  Generated from configure.in by autoheader.  */

/* Define if memory allocation logging and debugging is desired */
#undef DEBUG_ALLOCS

/* Define if libAfterBase is available */
#undef HAVE_AFTERBASE

/* Define if using builtin libjpeg */
#define  HAVE_BUILTIN_JPEG 1
#define HAVE_UNSIGNED_CHAR
#define HAVE_UNSIGNED_SHORT
#define HAVE_BOOLEAN
typedef unsigned char boolean;

/* Define if using builtin libpng */
#define HAVE_BUILTIN_PNG 1

/* Define to 1 if you have the <dirent.h> header file, and it defines `DIR'.
   */
#undef HAVE_DIRENT_H

/* Define to 1 if you have the <errno.h> header file. */
#define HAVE_ERRNO_H 1

/* Define if libFreeType is available  - should always be under win32 ! */
#define HAVE_FREETYPE 1

/* Define if libFreeType is available */
#define HAVE_FREETYPE_FREETYPE 1

/* Define to 1 if you have the <ft2build.h> header file. */
#define HAVE_FT2BUILD_H 1

/* Define if libgif is available */
#define HAVE_GIF

/* Define if using builtin libungif */
#define HAVE_BUILTIN_UNGIF 1

#if _MSC_VER >= 1400
#define NO_DOUBLE_FCLOSE_AFTER_FDOPEN
#else
/*#undef NO_DOUBLE_FCLOSE_AFTER_FDOPEN */
#endif

/* Define to 1 if you have the <inttypes.h> header file. */
#define HAVE_INTTYPES_H 1

/* Define if libjpeg is available */
#define HAVE_JPEG 1

/* Define if support for XPM images should be through libXpm */
#undef HAVE_LIBXPM

/* Define if support for XPM images should be through libXpm library in Xlib
   */
#undef HAVE_LIBXPM_X11

/* Define to 1 if you have the <malloc.h> header file. */
#define HAVE_MALLOC_H 1

/* Define to 1 if you have the <memory.h> header file. */
#define HAVE_MEMORY_H 1

/* Define if CPU supports MMX instructions */
#undef HAVE_MMX

/* Define to 1 if you have the <ndir.h> header file, and it defines `DIR'. */
#undef HAVE_NDIR_H

/* Define if libpng is available */
#define HAVE_PNG 1

/* We always use function prototypes - not supporting old compilers */
#define HAVE_PROTOTYPES 1

/* Define to 1 if you have the <stdarg.h> header file. */
#define HAVE_STDARG_H 1

/* Define to 1 if you have the <stddef.h> header file. */
#define HAVE_STDDEF_H 1

/* Define to 1 if you have the <stdint.h> header file. */
#undef HAVE_STDINT_H  /* VC++ does not have that ! */

/* Define to 1 if you have the <stdlib.h> header file. */
#define HAVE_STDLIB_H 1

/* Define to 1 if you have the <strings.h> header file. */
#undef HAVE_STRINGS_H 

/* Define to 1 if you have the <string.h> header file. */
#define HAVE_STRING_H 1

/* Define to 1 if you have the <sys/dirent.h> header file. */
#undef HAVE_SYS_DIRENT_H

/* Define to 1 if you have the <sys/dir.h> header file, and it defines `DIR'.
   */
#undef HAVE_SYS_DIR_H

/* Define to 1 if you have the <sys/ndir.h> header file, and it defines `DIR'.
   */
#undef HAVE_SYS_NDIR_H

/* Define to 1 if you have the <sys/stat.h> header file. */
#define HAVE_SYS_STAT_H 1

/* Define to 1 if you have the <sys/time.h> header file. */
#undef HAVE_SYS_TIME_H

/* Define to 1 if you have the <sys/types.h> header file. */
#define HAVE_SYS_TYPES_H 1

/* Define to 1 if you have the <sys/wait.h> header file. */
#define HAVE_SYS_WAIT_H 1

/* Define if libtiff is available */
#undef HAVE_TIFF

/* Define to 1 if you have the <unistd.h> header file. */
#undef HAVE_UNISTD_H

/* Define if support for XPM images is desired */
#define HAVE_XPM 1

/* Define to 1 if you have the <zlib.h> header file. */
#undef HAVE_ZLIB_H

/* Define if locale support in X is needed */
#undef I18N

/* Define to the address where bug reports for this package should be sent. */
#define PACKAGE_BUGREPORT "as-bugs@afterstep.org"

/* Define to the full name of this package. */
#define PACKAGE_NAME "libAfterImage"

/* Define to the full name and version of this package. */
#define PACKAGE_STRING "libAfterImage 0.99"

/* Define to the one symbol short name of this package. */
#define PACKAGE_TARNAME "libAfterImage.tar"

/* Define to the version of this package. */
#define PACKAGE_VERSION "0.99"

/* Support for shaped windows */
#undef SHAPE

/* Define to 1 if you have the ANSI C header files. */
#define STDC_HEADERS 1

/* Define to 1 if you can safely include both <sys/time.h> and <time.h>. */
#undef TIME_WITH_SYS_TIME

/* Define to 1 if your processor stores words with the most significant byte
   first (like Motorola and SPARC, unlike Intel and VAX). */
#undef WORDS_BIGENDIAN

/* Define if support for shared memory XImages is available */
#undef XSHMIMAGE

/* Define to 1 if the X Window System is missing or not being used. */
#define X_DISPLAY_MISSING 1

/* Define to 1 if type `char' is unsigned and you are not using gcc.  */
#ifndef __CHAR_UNSIGNED__
# undef __CHAR_UNSIGNED__
#endif

/* Define to empty if `const' does not conform to ANSI C. */
#undef const

/* Define as `__inline' if that's what the C compiler calls it, or to nothing
   if it is not supported. */
#define inline
