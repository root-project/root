// @(#)root/base:$Id$
// Author: Fons Rademakers   16/11/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#ifndef ROOT_TCanvasImp
#define ROOT_TCanvasImp

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TCanvasImp                                                           //
//                                                                      //
// ABC describing GUI independent main window (with menubar, scrollbars //
// and a drawing area).                                                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "Rtypes.h"

class TCanvas;
class TVirtualPadPainter;

class TCanvasImp {
friend class TCanvas;

protected:
   TCanvas  *fCanvas;   //TCanvas associated with this implementation

   TCanvasImp(const TCanvasImp& ci)
     : fCanvas(ci.fCanvas) { }
   TCanvasImp& operator=(const TCanvasImp& ci)
     {if(this!=&ci) fCanvas=ci.fCanvas; return *this;}

   virtual void   Lock();
   virtual void   Unlock() { }
   virtual Bool_t IsLocked() { return kFALSE; }

   virtual Bool_t PerformUpdate() { return kFALSE; }
   virtual TVirtualPadPainter *CreatePadPainter() { return 0; }

public:
   TCanvasImp(TCanvas *c=0) : fCanvas(c) { }
   TCanvasImp(TCanvas *c, const char *name, UInt_t width, UInt_t height);
   TCanvasImp(TCanvas *c, const char *name, Int_t x, Int_t y, UInt_t width, UInt_t height);
   virtual ~TCanvasImp() { }

   TCanvas       *Canvas() const { return fCanvas; }
   virtual void   Close() { }
   virtual void   ForceUpdate() { }
   virtual UInt_t GetWindowGeometry(Int_t &x, Int_t &y, UInt_t &w, UInt_t &h);
   virtual void   Iconify() { }
   virtual Int_t  InitWindow() { return 0; }
   virtual void   SetStatusText(const char *text = 0, Int_t partidx = 0);
   virtual void   SetWindowPosition(Int_t x, Int_t y);
   virtual void   SetWindowSize(UInt_t w, UInt_t h);
   virtual void   SetWindowTitle(const char *newTitle);
   virtual void   SetCanvasSize(UInt_t w, UInt_t h);
   virtual void   Show() { }
   virtual void   ShowMenuBar(Bool_t show = kTRUE);
   virtual void   ShowStatusBar(Bool_t show = kTRUE);
   virtual void   RaiseWindow();
   virtual void   ReallyDelete();

   virtual void   ShowEditor(Bool_t show = kTRUE);
   virtual void   ShowToolBar(Bool_t show = kTRUE);
   virtual void   ShowToolTips(Bool_t show = kTRUE);

   virtual Bool_t HasEditor() const { return kFALSE; }
   virtual Bool_t HasMenuBar() const { return kFALSE; }
   virtual Bool_t HasStatusBar() const { return kFALSE; }
   virtual Bool_t HasToolBar() const { return kFALSE; }
   virtual Bool_t HasToolTips() const { return kFALSE; }

   ClassDef(TCanvasImp,0)  //ABC describing main window protocol
};

#endif
