// Author: Sergey Linev, GSI  06/08/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TRaylibCanvas
#define ROOT_TRaylibCanvas

#include "TCanvasImp.h"
#include "TString.h"
#include <string>

class TTimer;

namespace ROOT {
namespace Experimental {

class TRaylibPadPainter;

/** \class TRaylibCanvas
    \ingroup raylibcanvas
    \brief TCanvasImp ABI implementation for raylib immediate-mode rendering

    Manages a single raylib window. Each canvas shares the same window
    (raylib limitation), with viewport switching via SetCameraMode().

    The render loop runs via a TTimer callback that polls input events
    and triggers BeginDrawing()/EndDrawing() frames.
*/
class TRaylibCanvas : public TCanvasImp {

protected:
   Int_t fWindowWidth = 0;      ///<! window width
   Int_t fWindowHeight = 0;     ///<! window height
   Int_t fPosX = 0;             ///<! window x position
   Int_t fPosY = 0;             ///<! window y position
   Bool_t fResized = kTRUE;     ///<! if window was resized
   Bool_t fMenuBar = kTRUE;     ///<! use of menu bar
   Bool_t fStatusBar = kTRUE;   ///<! use of status bar

   std::string fWindowTitle;    ///<! current window title

   TString fStatusMessage;
   int fFileMenuSelection = 0;
   bool fFileDropdownOpen = false;

   static void EnsureRaylibInitialized(int width, int height);

   Bool_t PerformUpdate(Bool_t async) override;
   TVirtualPadPainter *CreatePadPainter() override;

public:
   TRaylibCanvas(TCanvas *c, const char *name, Int_t x, Int_t y, UInt_t width, UInt_t height);
   ~TRaylibCanvas() override;

   // Window lifecycle
   Int_t InitWindow() override;
   void Close() override;
   void Show() override;

   // Geometry
   UInt_t GetWindowGeometry(Int_t &x, Int_t &y, UInt_t &w, UInt_t &h) override;
   void GetCanvasGeometry(Int_t wid, UInt_t &w, UInt_t &h) override;
   void ResizeCanvasWindow(Int_t wid) override;
   void UpdateDisplay(Int_t = 0, Bool_t = kFALSE) override;

   // UI elements (noop — raylib draws them manually if needed)
   void ShowMenuBar(Bool_t on = kTRUE) override { fMenuBar = on; fResized = kTRUE; }
   void ShowStatusBar(Bool_t on = kTRUE) override { fStatusBar = on; fResized = kTRUE; }
   void ShowEditor(Bool_t = kTRUE) override {}
   void ShowToolBar(Bool_t = kTRUE) override {}
   void ShowToolTips(Bool_t = kTRUE) override {}

   void ForceUpdate() override;

   void SetWindowPosition(Int_t x, Int_t y) override;
   void SetWindowSize(UInt_t w, UInt_t h) override;
   void SetWindowTitle(const char *newTitle) override;
   void SetCanvasSize(UInt_t w, UInt_t h) override;
   void Iconify() override;
   void RaiseWindow() override;

   Bool_t HasEditor() const override { return kFALSE; }
   Bool_t HasMenuBar() const override { return fMenuBar; }
   Bool_t HasStatusBar() const override { return fStatusBar; }
   Bool_t HasToolBar() const override { return kFALSE; }
   Bool_t HasToolTips() const override { return kFALSE; }

   void RunRaylib();

   // Static factory
   static TCanvasImp *NewCanvas(TCanvas *c, const char *name, Int_t x, Int_t y,
                                UInt_t width, UInt_t height);

   ClassDefOverride(TRaylibCanvas, 0) // raylib implementation for TCanvasImp
};

} // namespace Experimental
} // namespace ROOT

#endif