/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/*************************************************************************
* CompiledLib.h
*  This file contains interface definition of precompiled library. This
*  file has to be processed by cint with -c option in order to generate
*  a dictionary source. Do following to create G__clink.c and G__clink.h.
*
*     c:\>  cint -c-2 CompiledLib.h
*
*  Name of dictionary is G__clink.c , G__clink.h for C and
*  G__cpplink.cxx , G__cpplink.h for C++ by default. This can be changed by
*  -n option. Refer to doc/cint.txt , doc/makecint.txt and doc/ref.txt for
*  more detail.
*************************************************************************/

#include <windows.h>

void DrawRect4(HDC hdc);

