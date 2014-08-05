// @(#)root/qtgsi:$Id$
// Author: Denis Bertini  01/11/2000

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TQCanvasImp
#define ROOT_TQCanvasImp

//////////////////////////////////////////////////////////////////////////
//
// TQCanvasImp
//
// ABC describing Qt GUI independent main window (with menubar,
// scrollbars and a drawing area).
//
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TCanvasImp
#include "TCanvasImp.h"
#endif

class TQRootCanvas;

class TQCanvasImp :  public TCanvasImp {
protected:
   TQRootCanvas *fQCanvas; // Pointer to the Qt widget (TQRootCanvas)
   void Build(TCanvas *c, const char *name, Int_t x, Int_t y, UInt_t width, UInt_t height);
public:
   TQCanvasImp(TCanvas* = 0);
   TQCanvasImp(TCanvas *c, const char *name, UInt_t width, UInt_t height);
   TQCanvasImp(TCanvas *c, const char *name, Int_t x, Int_t y, UInt_t width, UInt_t height);
   virtual ~TQCanvasImp();

   ClassDef(TQCanvasImp,1)  //ABC describing Qt GUI independent main window
};

#endif

