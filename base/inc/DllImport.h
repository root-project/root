/* @(#)root/base:$Name:  $:$Id: DllImport.h,v 1.1.1.1 2000/05/16 17:00:39 rdm Exp $ */

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/*
  This include file defines the DllImport macro
  to build DLLs under Windows OS.

  They are defined as dummy for UNIX's
*/

#ifndef ROOT_DllImport
#define ROOT_DllImport

#ifndef __CINT__
#if defined(WIN32)
#  define R__DllImport  __declspec( dllimport )
#else
#  define R__DllImport
#endif
#  define R__EXTERN       R__DllImport extern
#endif


#endif
