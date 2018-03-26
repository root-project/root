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

ClassImp(TCanvasImp);
void TCanvasImp::Lock() {}
TCanvasImp::TCanvasImp(TCanvas *c, const char *, UInt_t, UInt_t) : fCanvas(c) { }
TCanvasImp::TCanvasImp(TCanvas *c, const char *, Int_t, Int_t, UInt_t, UInt_t) : fCanvas(c) { }
UInt_t TCanvasImp::GetWindowGeometry(Int_t &x, Int_t &y, UInt_t &w, UInt_t &h)
  { x = y = 0; w = h = 0; return 0;}
void TCanvasImp::SetStatusText(const char *, Int_t) { }
void TCanvasImp::SetWindowPosition(Int_t, Int_t) { }
void TCanvasImp::SetWindowSize(UInt_t, UInt_t) { }
void TCanvasImp::SetWindowTitle(const char *) { }
void TCanvasImp::SetCanvasSize(UInt_t, UInt_t) { }
void TCanvasImp::ShowMenuBar(Bool_t) { }
void TCanvasImp::ShowStatusBar(Bool_t) { }
void TCanvasImp::RaiseWindow() { }
void TCanvasImp::ReallyDelete() { }

void TCanvasImp::ShowEditor(Bool_t) { }
void TCanvasImp::ShowToolBar(Bool_t) { }
void TCanvasImp::ShowToolTips(Bool_t) { }
