/* @(#)root/zip:$Id$ */

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
/* tailor.h -- Not copyrighted 1993 Mark Adler */

/* Define MSDOS for Turbo C and Power C */
#ifdef WIN32
#define MSDOS
#endif
#ifdef __POWERC
#  define __TURBOC__
#  define MSDOS
#endif /* __POWERC */

#if (defined(__MSDOS__) && !defined(MSDOS))
#  define MSDOS
#endif

#ifdef ATARI_ST
#  undef MSDOS   /* avoid the MS-DOS specific includes */
#endif

/* Use prototypes and ANSI libraries if _STDC__, or Microsoft or Borland C,
 * or Silicon Graphics, or IBM C Set/2, or Watcom C, or GNU gcc under emx.
 */
#if defined(__STDC__) || defined(MSDOS) || defined(ATARI_ST) || defined(sgi)
#  ifndef PROTO
#    define PROTO
#  endif /* !PROTO */
#  define MODERN
#endif

#if defined(__IBMC__) || defined(__EMX__) || defined(__WATCOMC__) || defined(__MWERKS__)
#  ifndef PROTO
#    define PROTO
#  endif /* !PROTO */
#  define MODERN
#endif

#if defined(__BORLANDC__) || (defined(__alpha) && defined(VMS))
#  ifndef PROTO
#    define PROTO
#  endif /* !PROTO */
#  define MODERN
#endif

#if defined(__EMX__) || defined(__WATCOMC__) || defined(__BORLANDC__)
#  if (defined(OS2) && !defined(__32BIT__))
#    define __32BIT__
#  endif
#endif

#if (defined(__OS2__) && !defined(OS2))
#  define OS2
#endif

#ifdef __convexc__
#       define CONVEX
#endif /* __convexc__ */

#ifdef __COMPILER_KCC__
#  define TOPS20
#  define NOPROTO
#  define NO_SYMLINK
#  define NO_TERMIO
#  define DIRENT
#  define BIG_MEM
  extern int isatty();
# define R__window_size winsiz
#endif

/* Turn off prototypes if requested */
#if (defined(NOPROTO) && defined(PROTO))
#  undef PROTO
#endif

/* Used to remove arguments in function prototypes for non-ANSI C */
#ifdef PROTO
#  define OF(a) a
#else /* !PROTO */
#  define OF(a) ()
#endif /* ?PROTO */

/* Avoid using const if compiler does not support it */
#ifndef MODERN  /* if this fails, try: ifndef__STDC__ */
#  define const
#endif

#if defined(MACOS) || defined(__MWERKS__)
#  define DYN_ALLOC
#endif
#if (defined(MSDOS) && !defined(__GO32__) && !defined(WIN32))
#  ifdef __TURBOC__
#    include <alloc.h>
#    define DYN_ALLOC
     /* Turbo C 2.0 does not accept static allocations of large arrays */
     void far * fcalloc OF((unsigned items, unsigned size));
     void fcfree (void *ptr);
#  else /* !__TURBOC__ */
#    include <malloc.h>
#    define farmalloc _fmalloc
#    define farfree   _ffree
#    define fcalloc(nitems,itemsize) halloc((long)(nitems),(itemsize))
#    define fcfree(ptr) hfree((void huge *)(ptr))
#  endif /* ?__TURBOC__ */
#else /* !MSDOS */
#  if defined(WIN32)
#    include <malloc.h>
#  endif
#  ifdef __WATCOMC__
#    undef huge
#    undef far
#    undef near
#  else
#    define huge
#    define far
#    define near
#  endif
#  define farmalloc malloc
#  define farfree   free
#  define fcalloc(items,size) calloc((unsigned)(items), (unsigned)(size))
#  define fcfree    free
#  if (!defined(PROTO) && !defined(TOPS20))
     extern char *calloc(); /* essential for 16 bit systems (AT&T 6300) */
#  endif
#endif /* ?MSDOS */


#if (defined(OS2) && !defined(MSDOS))
/* MSDOS is defined anyway with MS C 16-bit. So the block above works.
 * For the 32-bit compilers, MSDOS must not be defined in the block above. */
#  define MSDOS
/* inherit MS-DOS file system etc. stuff */
#endif


/* Define MSVMS if MSDOS or VMS defined -- ATARI also does, Amiga could */
#if defined(MSDOS) || defined(VMS)
#  define MSVMS
#endif

/* case mapping functions. case_map is used to ignore case in comparisons,
 * to_up is used to force upper case even on Unix (for dosify option).
 */
 /* || defined(AMIGA) is removed from the next line because the line
    is too long for PATCHY */
#if defined(MSDOS) || defined(VMS) || defined(OS2) || defined(WIN32)
#  define case_map(c) upper[(c) & 0xff]
#  define to_up(c)    upper[(c) & 0xff]
#else
#  define case_map(c) (c)
#  define to_up(c)    ((c) >= 'a' && (c) <= 'z' ? (c)-'a'+'A' : (c))
#endif

/* Define void, voidp, and extent (size_t) */
#include <stdio.h>
#include <string.h>
#ifdef MODERN
#  if (!defined(M_XENIX) && !(defined(__GNUC__) && defined(sun)))
#    include <stddef.h>
#  endif /* !M_XENIX */
#  include <stdlib.h>
#  if defined(SYSV) || defined(__386BSD__)
#    include <unistd.h>
#  endif
   typedef size_t extent;
/* This definition of voidp is in conflict with the zlib one (zconf.h)
   voidp is used only in zlib code.
   typedef void voidp;
*/
#else /* !MODERN */
   typedef unsigned int extent;
#  define void int
/* This definition of voidp is in conflict with the zlib one (zconf.h)
   voidp is used only in zlib code.
   typedef char voidp;
*/
#endif /* ?MODERN */

/* Get types and stat */
#ifdef VMS
#  include <types.h>
#  include <stat.h>
#  define unlink delete
#  define NO_SYMLINK
#  define SSTAT vms_stat
#else /* !VMS */
#  if defined(MACOS)
#    include <types.h>
#    include <stddef.h>
#    include <Files.h>
#    include <StandardFile.h>
#    include <Think.h>
#    include <LoMem.h>
#    include <Pascal.h>
#    include "macstat.h"
#    define NO_SYMLINK
#  elif defined (__MWERKS__)
#    include <stddef.h>
#    include <stat.h>
#    define NO_SYMLINK
#  else
#    ifdef ATARI_ST
#      include <ext.h>
#      include <tos.h>
#    else
#      ifdef AMIGA
         int wild OF((char *));
         /* default to MEDIUM_MEM, but allow makefile override */
#        if ( (!defined(BIG_MEM)) && (!defined(SMALL_MEM)))
#           define MEDIUM_MEM
#        endif
#        if defined(LATTICE) || defined(__SASC)
#        include <sys/types.h>
#          include <sys/stat.h>
           extern int isatty(int);   /* SAS has no unistd.h */
#      endif
#        ifdef AZTEC_C
#          include "amiga/z-stat.h"
#          define RMDIR
#        endif
#      else /* !AMIGA */
#        include <sys/types.h>
#      include <sys/stat.h>
#    endif
#  endif
#  endif
#endif /* ?VMS */

/* Some systems define S_IFLNK but do not support symbolic links */
#if defined (S_IFLNK) && defined(NO_SYMLINK)
#  undef S_IFLNK
#endif


/* For Pyramid */
#ifdef pyr
#  define strrchr rindex
#  define ZMEM
#endif /* pyr */


/* File operations--use "b" for binary if allowed or fixed length 512 on VMS */
#ifdef VMS
#  define FOPR  "r","ctx=stm"
#  define FOPM  "r+","ctx=stm","rfm=fix","mrs=512"
#  define FOPW  "w","ctx=stm","rfm=fix","mrs=512"
#else /* !VMS */
#  if defined(MODERN)
#    define FOPR "rb"
#    define FOPM "r+b"
#    ifdef TOPS20 /* TOPS20 MODERN? You kidding? */
#      define FOPW "w8"
#    else
#      define FOPW "wb"
#    endif
#  else /* !MODERN */
#    ifdef AMIGA
#      define FOPR "rb"
#      define FOPM "rb+"
#      define FOPW "wb"
#    else /* !AMIGA */
#    define FOPR "r"
#    define FOPM "r+"
#    define FOPW "w"
#    endif /* ?AMIGA */
#  endif /* ?MODERN */
#endif /* VMS */

/* Open the old zip file in exclusive mode if possible (to avoid adding
 * zip file to itself).
 */
#ifdef OS2
#  define FOPR_EX FOPM
#else
#  define FOPR_EX FOPR
#endif

/* Define this symbol if your target allows access to unaligned data.
 * This is not mandatory, just a speed optimization. The compressed
 * output is strictly identical.
 */
#if (defined(MSDOS) && !defined(WIN32)) || defined(i386)
#    define UNALIGNED_OK
#endif
#if defined(mc68020) || defined(vax)
#    define UNALIGNED_OK
#endif

/* Under MSDOS we may run out of memory when processing a large number
 * of files. Compile with MEDIUM_MEM to reduce the memory requirements or
 * with SMALL_MEM to use as little memory as possible.
 */
#ifdef SMALL_MEM
#   define CBSZ 2048 /* buffer size for copying files */
#   define ZBSZ 2048 /* buffer size for temporary zip file */
#else
# ifdef MEDIUM_MEM
#   define CBSZ 8192
#   define ZBSZ 8192
# else
#  ifdef OS2
#   ifdef __32BIT__
#     define CBSZ 0x40000
#     define ZBSZ 0x40000
#   else
#     define CBSZ 0xE000
#     define ZBSZ 0x7F00 /* Some libraries do not allow a buffer size > 32K */
#   endif
#  else
#   ifdef TOPS20
#    define CBSZ 524288
#    define ZBSZ 524288
#   else
#    define CBSZ 16384
#    define ZBSZ 16384
#   endif
#  endif
# endif
#endif

#if (defined(BIG_MEM) || defined(MMAP)) && !defined(DYN_ALLOC)
#   define DYN_ALLOC
#endif

#ifdef __human68k__
#  include <sys/xglob.h>
#  define MSVMS
#  define SSTAT h68_stat
  int h68_stat OF((char *, struct stat *));
#  define OS_CODE  0x300  /* pretend it's Unix */
#endif

#ifdef ATARI_ST
#  define MSDOS  /* what? should be fixed */
#  define MSVMS
#  ifndef O_BINARY
#    define O_BINARY 0
#  endif
#  ifndef S_IFMT
#    define S_IFMT        (S_IFCHR|S_IFREG|S_IFDIR)
#  endif

#ifdef __IBMC__
#  ifndef S_IFMT
#    define S_IFMT 0xF000
#  endif
#endif /* __IBMC__ */

/* a whole bunch of functions needs Tos '\\' filenames
 * instead of '/',  the translation functions are in fileio.c:
 */
#  define unlink    st_unlink
#  define chmod     st_chmod
#  define mktemp    st_mktemp
#  define fopen     st_fopen
#  define open      st_open
#  define SSTAT     st_stat
#  define findfirst st_findfirst
#  define link      st_rename
#  define rmdir     st_rmdir

  int st_unlink    OF((char *));
  int st_chmod     OF((char *, int));
  char *st_mktemp  OF((char *));
  FILE *st_fopen   OF((char *, char *));
  int st_open      OF((char *, int));
  int st_stat      OF((char *, struct stat *));
  int st_findfirst OF((char *, struct ffblk *, int));
  int st_rename    OF((char *, char *));
  int st_rmdir     OF((char *));
#endif  /* ATARI */

#ifndef SSTAT
#  define SSTAT      stat
#endif
#ifdef S_IFLNK
#  define LSTAT      lstat
#else
#  define LSTAT      SSTAT
#endif


/* The following OS codes are defined in pkzip appnote.txt */
#ifdef AMIGA
#  define OS_CODE  0x100
#endif
#ifdef VMS
#  define OS_CODE  0x200
#endif
/* unix    3 */
/* vms/cms 4 */
#ifdef ATARI_ST
#  define OS_CODE  0x500
#endif
#ifdef OS2
#  define OS_CODE  0x600
#endif
#if defined(MACOS) || defined(__MWERKS__)
#  define OS_CODE  0x700
#endif
/* z system 8 */
/* cp/m     9 */
#ifdef TOPS20
#  define OS_CODE  0xa00
#endif
#ifdef WIN32
#  define OS_CODE  0xb00
#endif
/* qdos 12 */

#if defined(MSDOS) && !defined(OS_CODE)
#  define OS_CODE  0x000
#endif
#ifndef OS_CODE
#  define OS_CODE  0x300  /* assume Unix */
#  ifndef R__UNIX
#    define R__UNIX
#  endif
#endif


/* end of tailor.h */

