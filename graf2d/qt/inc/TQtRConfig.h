// Author: Valeri Fine   28/06/2004
/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * Copyright (C) 2004 by Valeri Fine.                                    *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TQTRCONFIG
#define ROOT_TQTRCONFIG
#include "RConfig.h"
#ifndef __CINT__
#  include "qglobal.h"
#endif

// defined the CPP symbols to create the different versions
#ifdef R__QTX11
#undef R__QTX11
#endif /* R__QTX11 */

#ifdef   R__QTWIN32
#undef   R__QTWIN32
#endif /*R__QTWIN32*/

#ifdef   R__QTMACOS
#undef   R__QTMACOS
#endif /*R__QTMACOS*/

#ifdef  R__QTGUITHREAD
#undef  R__QTGUITHREAD
#endif /*R__QTGUITHREAD*/

#ifdef Q_WS_X11
# define R__QTX11
#endif

#if !defined(R__QTX11) && defined(R__UNIX) && !defined(R__MACOSX)
# define R__QTX11
#endif

#if !defined(R__QTX11) &&  defined(R__WIN32)
# define R__QTWIN32
#endif

#if defined(R__MACOSX)
# define R__QTMACOS
#endif

#ifndef ROOT_VALID_QT_VERSION
// The least Qt version the ROOT package can work with
#  ifndef __CINT__
#    if QT_VERSION < 0x40000
#      define ROOT_VALID_QT_VERSION "3.2.0"
#    else
#      define ROOT_VALID_QT_VERSION "4.1.2"
#    endif
#  endif
#endif

#endif /*ROOT_TQTCONFIG*/
