/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/*************************************************************************
* lib/xlib/XLIB.h
*  header file to link Xlib and cint.
*  Date     6 Oct 1996   
*  Copyright(c)    1996    Masaharu Goto
*************************************************************************/

#ifdef G__APPLE
#include "X11/Xresource.h"
#endif

#ifdef __MAKECINT__
#include <platform.h>
#endif

#include "X11/Xlib.h"
#include "X11/Xutil.h"
#include "X11/Xos.h"
#include "X11/keysym.h"
