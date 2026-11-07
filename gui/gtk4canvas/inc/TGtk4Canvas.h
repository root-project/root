// Author: Sergey Linev, GSI   12/08/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGtk4Canvas
#define ROOT_TGtk4Canvas

#include "TCanvasImp.h"

class Gtk4CanvasWindow;
class Gtk4DrawArea;

namespace ROOT {
namespace Experimental {


class TGtk4Canvas : public TCanvasImp {

protected:

   Gtk4CanvasWindow *fCanvasWindow = nullptr;
   Gtk4DrawArea *fDrawArea = nullptr;

   Bool_t fFixedSize = kFALSE;      ///<! true when fixed-size canvas is configured
   UInt_t fClientBits = 0;          ///<! latest status bits from client like editor visible or not

   Bool_t PerformUpdate(Bool_t async) override;
   TVirtualPadPainter *CreatePadPainter() override;

   void AssignStatusBits(UInt_t bits);


public:
   TGtk4Canvas(TCanvas *c, const char *name, Int_t x, Int_t y, UInt_t width, UInt_t height);
   ~TGtk4Canvas() override;

   Gtk4DrawArea *GetDrawArea() const { return fDrawArea; }

   Int_t InitWindow() override;
   void Close() override;
   void Show() override;

   UInt_t GetWindowGeometry(Int_t &x, Int_t &y, UInt_t &w, UInt_t &h) override;
   void GetCanvasGeometry(Int_t wid, UInt_t &w, UInt_t &h) override;
   void ResizeCanvasWindow(Int_t) override {}
   void UpdateDisplay(Int_t = 0, Bool_t = kFALSE) override {}

   void ShowMenuBar(Bool_t = kTRUE) override { }
   void ShowStatusBar(Bool_t = kTRUE) override { }
   void ShowEditor(Bool_t = kTRUE) override {  }
   void ShowToolBar(Bool_t = kTRUE) override { }
   void ShowToolTips(Bool_t  = kTRUE) override { }

   void ForceUpdate() override;

   void SetWindowPosition(Int_t x, Int_t y) override;
   void SetWindowSize(UInt_t w, UInt_t h) override;
   void SetWindowTitle(const char *newTitle) override;
   void SetCanvasSize(UInt_t w, UInt_t h) override;
   void SetCursor(ECursor cursor) override;
   void Iconify() override;
   void RaiseWindow() override;

   Bool_t HasEditor() const override;
   Bool_t HasMenuBar() const override;
   Bool_t HasStatusBar() const override;
   Bool_t HasToolBar() const override { return kFALSE; }
   Bool_t HasToolTips() const override;

   static TCanvasImp *NewCanvas(TCanvas *c, const char *name, Int_t x, Int_t y, UInt_t width, UInt_t height);

   ClassDefOverride(TGtk4Canvas, 0) // Gtk4 implementation for TCanvasImp
};

} // namespace Experimental
} // namespace ROOT

#endif
