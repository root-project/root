/* @(#)root/base:$Name$:$Id$ */

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/*
  This include file defines DllImport/DllExport macro
  to build DLLs under Windows OS.

  They are defined as dummy for UNIX's
*/

#ifndef ROOT_DllImport
#define ROOT_DllImport

#ifndef __CINT__
#if defined(WIN32)
#  define DllImport     __declspec( dllimport )
#else
#  define DllImport
#endif
#if defined(_DLL)
#  define DllExport     __declspec( dllexport )
#else
#  define DllExport
#endif
#  define R__EXTERN       DllImport extern
#endif


#endif
