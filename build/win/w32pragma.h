/* @(#)build/win:$Name:$:$Id:$ */

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
 * Some pragma's to turn off some annoying VC++ 6 warnings.              *
 *                                                                       *
 *************************************************************************/

#ifdef _WIN32
    /* Disable warning about truncated symboles (usually coming from stl) */
#   pragma warning (disable: 4786)
    /* Disable warning about inconsistent dll linkage (dllexport assumed) */
#   pragma warning (disable: 4273)
#endif

#endif

