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
#include "TSystem.h"


////////////////////////////////////////////////////////////////////////////////
/// Returns true if gVirtualX instance used by the canvas implementation
/// In the ROOT6 it is be default on - means all necessary libraries will be loaded and initialized
/// In the ROOT7 it will be switched to off by default and probably completely removed
/// from the API

Bool_t TCanvasImp::IsUseVirtualX() const
{
   return kTRUE;
}

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

////////////////////////////////////////////////////////////////////////////////
/// Update canvas display, also optionally sleep to wait until operation finished

void TCanvasImp::UpdateDisplay(Int_t /* mode */, Bool_t /* sleep */)
{
}
