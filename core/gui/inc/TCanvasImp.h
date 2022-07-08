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
   TCanvas  *fCanvas{nullptr};   //TCanvas associated with this implementation

   TCanvasImp(const TCanvasImp &ci) : fCanvas(ci.fCanvas) {}
   TCanvasImp &operator=(const TCanvasImp &ci)
   {
      if (this != &ci)
         fCanvas = ci.fCanvas;
      return *this;
   }

   virtual void   Lock() {}
   virtual void   Unlock() {}
   virtual Bool_t IsLocked() { return kFALSE; }

   virtual Bool_t IsWeb() const { return kFALSE; }
   virtual Bool_t PerformUpdate() { return kFALSE; }
   virtual TVirtualPadPainter *CreatePadPainter() { return nullptr; }

public:
   TCanvasImp(TCanvas *c = nullptr) : fCanvas(c) {}
   TCanvasImp(TCanvas *c, const char *name, UInt_t width, UInt_t height) : fCanvas(c) { (void) name; (void) width; (void) height; }
   TCanvasImp(TCanvas *c, const char *name, Int_t x, Int_t y, UInt_t width, UInt_t height) : fCanvas(c) {(void) name; (void) x; (void) y; (void) width; (void) height;}
   virtual ~TCanvasImp() {}

   TCanvas       *Canvas() const { return fCanvas; }
   virtual void   Close() {}
   virtual void   ForceUpdate() {}
   virtual UInt_t GetWindowGeometry(Int_t &x, Int_t &y, UInt_t &w, UInt_t &h)
   {
      x = y = 0;
      w = h = 0;
      return 0;
   }
   virtual void Iconify() {}
   virtual Int_t  InitWindow() { return 0; }
   virtual void   SetStatusText(const char *text = nullptr, Int_t partidx = 0) { (void) text; (void) partidx; }
   virtual void   SetWindowPosition(Int_t x, Int_t y) { (void) x; (void) y; }
   virtual void   SetWindowSize(UInt_t width, UInt_t height) { (void) width; (void) height; }
   virtual void   SetWindowTitle(const char *newTitle) { (void) newTitle; }
   virtual void   SetCanvasSize(UInt_t w, UInt_t h) { (void) w; (void) h; }
   virtual void   Show() {}
   virtual void   ShowMenuBar(Bool_t show = kTRUE) { (void) show; }
   virtual void   ShowStatusBar(Bool_t show = kTRUE) { (void) show; }
   virtual void   RaiseWindow() {}
   virtual void   ReallyDelete() {}

   virtual void   ShowEditor(Bool_t show = kTRUE) { (void) show; }
   virtual void   ShowToolBar(Bool_t show = kTRUE) { (void) show; }
   virtual void   ShowToolTips(Bool_t show = kTRUE) { (void) show; }

   virtual Bool_t HasEditor() const { return kFALSE; }
   virtual Bool_t HasMenuBar() const { return kFALSE; }
   virtual Bool_t HasStatusBar() const { return kFALSE; }
   virtual Bool_t HasToolBar() const { return kFALSE; }
   virtual Bool_t HasToolTips() const { return kFALSE; }

   ClassDef(TCanvasImp,0)  //ABC describing main window protocol
};

#endif
