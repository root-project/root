// @(#)root/base:$Id$
// Author: Fons Rademakers   16/11/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TCanvasImp
\ingroup Base

ABC describing GUI independent main window (with menubar, scrollbars
and a drawing area).
*/

#include "TCanvasImp.h"
#include "TVirtualX.h"


////////////////////////////////////////////////////////////////////////////////
/// Change mouse pointer, redirect to gVirtualX

void TCanvasImp::Warp(Int_t ix, Int_t iy)
{
   if(gVirtualX)
      gVirtualX->Warp(ix, iy);
}

////////////////////////////////////////////////////////////////////////////////
/// Request current mouse pointer, redirect to gVirtualX

Int_t TCanvasImp::RequestLocator(Int_t &x, Int_t &y)
{
   return gVirtualX ? gVirtualX->RequestLocator(1, 1, x, y) : -1;
}

////////////////////////////////////////////////////////////////////////////////
/// Gets the size and position of the canvas paint area.

void TCanvasImp::GetCanvasGeometry(Int_t wid, UInt_t &w, UInt_t &h)
{
   Int_t x, y;
   gVirtualX->GetGeometry(wid, x, y, w, h);
}

////////////////////////////////////////////////////////////////////////////////
/// Resize canvas window, redirect to gVirtualX

void TCanvasImp::ResizeCanvasWindow(Int_t wid)
{
   if (gVirtualX)
      gVirtualX->ResizeWindow(wid);   //resize canvas and off-screen buffer
}
