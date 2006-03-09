/* @(#)build/win:$Name:  $:$Id: w32pragma.h,v 1.1 2002/05/03 18:13:32 rdm Exp $ */

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_w32pragma
#define ROOT_w32pragma

/*************************************************************************
 *                                                                       *
 * w32pragma                                                             *
 *                                                                       *
 * Pragmas and defines for MSVC                                          *
 *                                                                       *
 *************************************************************************/

#ifdef _WIN32

/* Disable warning about truncated symboles (usually coming from stl) */
#pragma warning (disable: 4786)
/* Disable warning about inconsistent dll linkage (dllexport assumed) */
#pragma warning (disable: 4273)
/* "no suitable definition provided for explicit template instantiation"*/
#pragma warning (disable: 4661)
/* "deprecated, use ISO C++ conformant name" */
#pragma warning (disable: 4996)

#define WIN32 1
#define _WINDOWS 1
#define WINVER 0x0400
#define CRTAPI1 _cdecl 
#define CRTAPI2 _cdecl
#define _X86_ 1 
// #define _DLL  - used to be explicitely defined, 
// but it's implicitely defined via /MD(d)
#define VISUAL_CPLUSPLUS 1
#define G__REDIRECTIO 1
#define G__SHAREDLIB 1
#define G__UNIX 1
#define G__ROOT 1
#define G__WIN32 1

#if (_MSC_VER >= 1310)
#  define G__NEWSTDHEADER 1
#endif

#if (_MSC_VER >= 1400)
#define _CRT_SECURE_NO_DEPRECATE
#define _SECURE_SCL 0
#define _HAS_ITERATOR_DEBUGGING 0
#endif

#endif // _WIN32

#endif // defined ROOT_w32pragma
