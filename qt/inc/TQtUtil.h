// @(#)root/qt:$Name:$:$Id:$
// Author: Valeri Fine   21/01/2002

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * Copyright (C) 2002 by Valeri Fine.                                    *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TQtUtil
#define ROOT_TQtUtil

#include "TGQt.h"
#include "TCanvas.h"
#include "TCanvasImp.h"

#include "qpixmap.h"
#include "qwidget.h"

//----------------------------------------
//      Q: How to get Qt pointer:
//----------------------------------------

//_______________________________________
inline QPixmap *padPixmap(TPad *pad)
{     return (QPixmap *)TGQt::iwid(pad->GetPixmapID());   }
//_______________________________________
inline QWidget *canvasWidget(TCanvas *c)
{  return (QWidget *)TGQt::iwid(c->GetCanvasID()) ; }
//_______________________________________
inline QWidget *canvasWidget(TCanvasImp *c)
{ return (QWidget *) TGQt::iwid(((TQtCanvasImp *)c)->GetCanvasImpID()); }
//_______________________________________
inline QWidget *mainWidget(TCanvas *c)
{  return canvasWidget(c->GetCanvasImp());}

//----------------------------------------
// Q: Get WIN32/X11 handles:
//    (see function above and Qt manual also)
//----------------------------------------
//_______________________________________
inline HDC wigdetHdc(TPad *pad)
{  return padPixmap(pad)->handle(); }
//_______________________________________
inline HDC wigdetHdc(TCanvas *c)
{ return canvasWidget(c)->handle(); }
//_______________________________________
inline HDC wigdetHdc(TCanvasImp *c)
{ return canvasWidget(c)->handle(); }

ifdef WIN32
//_______________________________________
inline HWND hwndWin32(TCanvas *c)
{  return canvasWidget(c)->winId(); }
//_______________________________________
inline HWND hwndWin32(TCanvasImp *c)
{ return canvasWidget(c)->winId(); }
#else
//_______________________________________
inline Ulong_t hwndWin32(TCanvas *c)
{  return canvasWidget(c)->winId(); }
//_______________________________________
inline Ulong_t hwndWin32(TCanvasImp *c)
{ return canvasWidget(c)->winId(); }
#endif

#endif

