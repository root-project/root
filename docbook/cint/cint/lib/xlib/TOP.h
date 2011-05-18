/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/***********************************************************************
* lib/xlib/TOP.h
*  header file to link xlib and cint.
*  Date     6 Oct 1996   
*  Copyright(c)    1996    Masaharu Goto
*
*  Top level parameter information file for making cintxlib
***********************************************************************/

/***********************************************************************
* X11 include files are included in XLIB.h. XLIB.h and X11 include files
* are preprocessed by external C preprocessor. All of the macro 
* information is lost in this process. 
***********************************************************************/

#ifdef __MAKECINT__
#pragma preprocessor on
#endif

#include "XLIB.h"

#ifdef __MAKECINT__
#pragma preprocessor off
#endif

#define G__XLIBDLL_H

/***********************************************************************
* Following part is added and exposed to interpreter, because TOP.h is
* not preprocessed.
***********************************************************************/
#ifdef __MAKECINT__
#include "x11const.h"
#include "x11mfunc.h"

#pragma link off class _XPrivate;
#pragma link off class _XrmHashBucketRec;
#pragma link off class _XOM;
#pragma link off class _XOC;
#pragma link off class _XIM;
#pragma link off class _XIC;
#pragma link off class _XRegion;
//#pragma link off function gettimeofday; /* RH7.0 warning */
//#pragma link off function XStringToContext;
//#pragma link off function XUniqueContext;

#endif

