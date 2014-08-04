// @(#)root/qt:$Id$
// Author: Valeri Fine   21/01/2002

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * Copyright (C) 2002 by Valeri Fine.                                    *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TQtUTIL
#define ROOT_TQtUTIL

#include "TGQt.h"
#include "TVirtualPad.h"
#include "TCanvasImp.h"

#include <QtGui/QPixmap>
#include <QtGui/QWidget>

//----------------------------------------
//      Q: How to get Qt pointer:
//----------------------------------------
namespace  TQtUtil {
///
/// Returns QPixmap backend for the given TVirtualPad
///
//_______________________________________
inline QPixmap *padPixmap(TVirtualPad *pad)
{     return (QPixmap *)TGQt::iwid(pad->GetPixmapID());   }

///
/// Returns QWidget backend for the given TCanvas
///  if "c" is not a TCanvas returns zero
///
//_______________________________________
inline QWidget *canvasWidget(TVirtualPad *c)
{  return (QWidget *)TGQt::iwid(c->GetCanvasID()) ; }
//----------------------------------------
// Q: Get WIN32/X11 handles:
//    (see function above and Qt manual also)
//----------------------------------------
///
/// Returns system depended backend handle
/// for the given TVirtualPad
///
//_______________________________________
inline unsigned long  wigdetHdc(TVirtualPad *pad)
{  return padPixmap(pad)->handle(); }

///
/// Returns system depended backend handle
/// for the given TCanvas
///  if "c" is not a TCanvas returns zero

//_______________________________________
inline unsigned long  hwndWin32(TVirtualPad *c)
{  return canvasWidget(c)->winId(); }
};
#endif

