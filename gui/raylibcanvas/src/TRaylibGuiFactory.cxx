// Author: Sergey Linev, GSI  06/08/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TRaylibGuiFactory.h"
#include "TRaylibCanvas.h"

using namespace ROOT::Experimental;

/** \class TRaylibGuiFactory
    \ingroup raylibcanvas

    Factory for ROOT GUI components using raylib as the rendering backend.
*/

// ─── Constructor ──────────────────────────────────────────────────────

TRaylibGuiFactory::TRaylibGuiFactory(const char *name, const char *title)
   : TGuiFactory(name, title)
{
}

// ─── Create Canvas Imp (2 overloads) ──────────────────────────────────

TCanvasImp *TRaylibGuiFactory::CreateCanvasImp(TCanvas *c, const char *title,
                                                UInt_t width, UInt_t height)
{
   return TRaylibCanvas::NewCanvas(c, title, -1, -1, width, height);
}

TCanvasImp *TRaylibGuiFactory::CreateCanvasImp(TCanvas *c, const char *title,
                                                Int_t x, Int_t y, UInt_t width, UInt_t height)
{
   return TRaylibCanvas::NewCanvas(c, title, x, y, width, height);
}

