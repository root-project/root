/* @(#)root/base:$Name:  $:$Id: RConfig.h,v 1.43 2002/07/11 21:32:35 rdm Exp $ */

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RConfig
#define ROOT_RConfig

/*************************************************************************
 *                                                                       *
 * RConfig                                                               *
 *                                                                       *
 * Defines used by ROOT.                                                 *
 *                                                                       *
 *************************************************************************/

#ifndef ROOT_RVersion
#include "RVersion.h"
#endif


/*---- new C++ features ------------------------------------------------------*/

#define R__RTTI
#define R__USE_SHADOW_CLASS

/*---- machines --------------------------------------------------------------*/

#ifdef __hpux
#   ifdef __ia64
#      define R__HPUX11    /* find a better test for HP-UX 11 */
#   endif
#   define R__HPUX
#   define R__UNIX
#   define ANSICPP
#   ifdef __LP64__
#      define R__B64
#   endif
#   ifndef R__HPUX11
#      define NEED_SNPRINTF
#   endif
#endif

#ifdef _AIX
#   define R__AIX
#   define R__UNIX
#   define ANSICPP
#   define NEED_STRCASECMP
#endif

#ifdef __linux
#   ifndef linux
#      define linux
#   endif
#endif

#if defined(__alpha) && !defined(linux)
#   include <standards.h>
#   ifdef _XOPEN_SOURCE
#      if _XOPEN_SOURCE+0 > 0
#         define R__TRUE64
#      endif
#   endif
#   define R__ALPHA
#   define ANSICPP
#   ifndef R__TRUE64
#      define NEED_SNPRINTF
#   endif
#   ifndef __VMS
#      define R__UNIX
#      define R__B64
#      define R__BYTESWAP
#      if __DECCXX_VER >= 60060002
#         define R__VECNEWDELETE /* supports overloading of new[] and delete[] */
#         define R__PLACEMENTDELETE /* supports overloading placement delete */
#         define R__PLACEMENTINLINE /* placement new/delete is inline in <new> */
#         define R__THROWNEWDELETE  /* new/delete throw exceptions */
#      endif
#      if defined __GNUC__
#         define R__NAMESPACE_TEMPLATE_IMP_BUG
#      else
#         define R__TEMPLATE_OVERLOAD_BUG
#      endif
#   else
#      define R__VMS
#      define cxxbug
#      define NEED_STRCASECMP
#      define R__NONSCALARFPOS
#   endif
#endif

#if defined(__sun) && !defined(linux)
#   ifdef __SVR4
#      define R__SOLARIS
#      define ANSICPP
#      ifdef __i386
#         define R__I386
#         define R__BYTESWAP
#      endif
#   else
#      define R__SUN
#      include <stdlib.h>
#   endif
#   define R__UNIX
#   define NEED_STRING
#   define NEED_SIGJMP
#   if __SUNPRO_CC > 0x420
#      define R__SOLARIS_CC50
#      define R__ANSISTREAM      /* ANSI C++ Standard Library conformant */
#   endif
#   if __SUNPRO_CC >= 0x420
#      define R__SUNCCBUG        /* to work around a compiler bug */
#   endif
#endif

#if defined(__sgi) && !defined(linux)
#   define R__SGI
#   define R__UNIX
#   define ANSICPP
#   define NEED_STRING
#   define NEED_SIGJMP
#   ifdef IRIX64
#      define R__SGI64
#   endif
#endif

#if defined(linux)
#   include <features.h>
#   if __GNU_LIBRARY__ == 6
#      ifndef R__GLIBC
#         define R__GLIBC
#      endif
#   endif
#   if __GLIBC__ == 2 && __GLIBC_MINOR__ >= 2
#      define R__NONSCALARFPOS2
#      define R__USESTHROW
#   endif
#endif

#if defined(linux) && defined(__i386__)
#   define R__LINUX
#   define R__UNIX
#   define R__BYTESWAP
#   ifndef __i486__
#      define __i486__       /* turn off if you really want to run on an i386 */
#   endif
#   define NEED_SIGJMP
#endif

#if defined(linux) && defined(__ia64__)
#   define R__LINUX
#   define R__UNIX
#   define R__BYTESWAP
#   define R__B64
#   define NEED_SIGJMP
#endif

#if defined(linux) && defined(__alpha__)
#   define R__LINUX
#   define R__UNIX
#   define R__BYTESWAP
#   define R__B64
#   define NEED_SIGJMP
#endif

#if defined(linux) && defined(__arm__)
#   define R__LINUX
#   define R__UNIX
#   define R__BYTESWAP
#   define NEED_SIGJMP
#endif

#if defined(linux) && defined(__sparc__)
#   define R__LINUX
#   define R__UNIX
#   define NEED_SIGJMP
/*#   define R__B64 */     /* enable when 64 bit machine */
#endif

#if defined(linux) && defined(__sgi)
#   define R__LINUX
#   define R__UNIX
#   define NEED_SIGJMP
/*#   define R__B64 */     /* enable when 64 bit machine */
#endif

#if defined(linux) && defined(__powerpc__)
#   define R__MKLINUX
#   define R__LINUX
#   define R__UNIX
#   define NEED_SIGJMP
#   if __GNUC__ >= 3 || __GNUC_MINOR__ >= 90   /* modern egcs/gcc */
#      define R__PPCEGCS
#   endif
#endif

#if defined(__MACH__) && defined(__i386__)
#   define R__HURD
#   define f2cFortran   /* cfortran.h does not know HURD - sigh */
#   define R__UNIX
#   define R__BYTESWAP
#   define R__GLIBC     /* GNU/Hurd always use GLIBC 2.x :-) */
#   define NEED_SIGJMP
#endif

#if defined(__Lynx__) && defined(__powerpc__)
#   define R__LYNXOS
#   define R__UNIX
#   define ANSICPP
#   define NEED_SIGJMP
#   define NEED_STRCASECMP
#   define NEED_SNPRINTF
#endif

#if defined(__FreeBSD__)
#   define R__FBSD
#   define R__UNIX
#   define R__BYTESWAP
#   define R__NOSTATS      /* problem using stats with FreeBSD malloc/free */
#endif

#if defined(__APPLE__)     /* MacOS X support, initially following FreeBSD */
#   define R__MACOSX
#   define R__UNIX
#endif

#ifdef _HIUX_SOURCE
#   define R__HIUX
#   define R__UNIX
#   define NEED_SIGJMP
#   define NEED_SNPRINTF
#   define ANSICPP
#endif

#ifdef __GNUC__
#   define R__GNU
#   define ANSICPP
#   if __GNUC__ >= 3 || __GNUC_MINOR__ >= 90    /* egcs 1.0.3 */
#      define R__VECNEWDELETE    /* supports overloading of new[] and delete[] */
#      define R__PLACEMENTDELETE /* supports overloading placement delete */
#   endif
#   if __GNUC__ >= 3 || __GNUC_MINOR__ >= 91    /* egcs 1.1.x */
#      define R__ANSISTREAM      /* ANSI C++ Standard Library conformant */
#   endif
#   if __GNUC__ >= 3 && __GNUC_MINOR__ >=0 && __GNUC_MINOR__ < 8
#      define R__SSTREAM         /* strstream renamed to sstream */
#   endif
#   if defined(__ia64__) &&  __GNUC__ < 3       /* gcc 2.9x (MINOR is 9!) */
#      define R__VECNEWDELETE    /* supports overloading of new[] and delete[] */
#      define R__PLACEMENTDELETE /* supports overloading placement delete */
#      define R__ANSISTREAM      /* ANSI C++ Standard Library conformant */
#   endif
#endif

#ifdef __KCC
#   define R__KCC
#   define R__ANSISTREAM      /* ANSI C++ Standard Library conformant */
#   define R__VECNEWDELETE    /* supports overloading of new[] and delete[] */
#   define R__PLACEMENTDELETE /* supports overloading placement delete */
#   define R__PLACEMENTINLINE /* placement new/delete is inline in <new> */
#   define ANSICPP
#endif

#ifdef __INTEL_COMPILER
#   define R__INTEL_COMPILER
#   define R__ANSISTREAM      /* ANSI C++ Standard Library conformant */
#   define R__VECNEWDELETE    /* supports overloading of new[] and delete[] */
#   define R__PLACEMENTDELETE /* supports overloading placement delete */
#   define ANSICPP
#endif

#ifdef __HP_aCC
#   define R__ACC
#   define R__VECNEWDELETE    /* supports overloading of new[] and delete[] */
#   if __HP_aCC >= 53000
#      define R__PLACEMENTDELETE /* supports overloading placement delete */
#      define R__PLACEMENTINLINE /* placement new/delete is inline in <new> */
#      define R__THROWNEWDELETE  /* new/delete throw exceptions */
#      define R__ANSISTREAM      /* ANSI C++ Standard Library conformant */
#      define R__TMPLTSTREAM     /* iostream implemented with templates */
#   else
#      define R__TEMPLATE_OVERLOAD_BUG
#      define R__GLOBALSTL       /* STL in global name space */
#   endif
#endif

#ifdef _WIN32
#   define R__WIN32
#   ifndef WIN32
#      define WIN32
#   endif
#   define R__BYTESWAP
#   define R__ACCESS_IN_SYMBOL
#endif


#ifdef BORLAND
#   define MSDOS        /* Windows will always stay MSDOS */
#   define ANSICPP
#   define R__INT16
#   define R__BYTESWAP
#endif

#ifdef __SC__
#   define SC
#   define R__SC
#   if defined(macintosh)
#      define R__MAC
#      define NEED_STRING
#      define ANSICPP
#   elif WIN32
#      define NEED_STRING
#      define NEED_STRCASECMP
#      define NEED_SNPRINTF
#      define ANSICPP
#   else
#      define MSDOS
#      define NEED_STRCASECMP
#      define R__BYTESWAP
#   endif
#endif

#ifdef _MSC_VER
#   define R__VISUAL_CPLUSPLUS
#   define NEED_STRING
#   define NEED_STRCASECMP
#   define NEED_SNPRINTF
#   define ANSICPP
#   define R__VECNEWDELETE    /* supports overloading of new[] and delete[] */
#   define R__PLACEMENTDELETE /* supports overloading placement delete */
#endif

#ifdef __MWERKS__
#   define R__MWERKS
#   define R__MAC
#   define ANSICPP
#   define NEED_STRING
#   define NEED_STRCASECMP
#   define NEED_SNPRINTF
#endif


/*--- memory and object statistics -------------------------------------------*/

/* #define R__NOSTATS */


/*--- cpp --------------------------------------------------------------------*/

#ifdef ANSICPP
    /* symbol concatenation operator */
#   define _NAME1_(name) name
#   define _NAME2_(name1,name2) name1##name2
#   define _NAME3_(name1,name2,name3) name1##name2##name3

    /* stringizing */
#   define _QUOTE_(name) #name

#else

#   define _NAME1_(name) name
#   define _NAME2_(name1,name2) _NAME1_(name1)name2
#   define _NAME3_(name1,name2,name3) _NAME2_(name1,name2)name3

#   define _QUOTE_(name) "name"

#endif

/* produce an identifier that is almost unique inside a file */
#ifndef __CINT__
#   define _R__JOIN_(X,Y) _NAME2_(X,Y)
#   define _R__UNIQUE_(X) _R__JOIN_(X,__LINE__)
#else
    /* Currently CINT does not really mind to have duplicates and     */
    /* does not work correctly as far as merging tokens is concerned. */
#   define _R__UNIQUE_(X) X
#endif

/*---- misc ------------------------------------------------------------------*/

#ifdef R__GNU
#   define SafeDelete(p) { if (p) { delete p; p = 0; } }
#else
#   define SafeDelete(p) { delete p; p = 0; }
#endif

#endif

