// Author: Sergey Linev, GSI   26/06/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TQt6Canvas
#define ROOT_TQt6Canvas

#include "TCanvasImp.h"

class TPad;
class TExec;
class QWidget;

class TQt6Canvas : public TCanvasImp {

protected:

   QWidget *fWidget = nullptr;

   Bool_t fFixedSize = kFALSE;      ///<! true when fixed-size canvas is configured
   UInt_t fClientBits = 0;          ///<! latest status bits from client like editor visible or not

   void ProcessExecs(TPad *pad, TExec *extra = nullptr);

   Bool_t PerformUpdate(Bool_t async) override;
   TVirtualPadPainter *CreatePadPainter() override;

   void AssignStatusBits(UInt_t bits);


public:
   TQt6Canvas(TCanvas *c, const char *name, Int_t x, Int_t y, UInt_t width, UInt_t height);
   ~TQt6Canvas() override;

   Int_t InitWindow() override;
   void Close() override;
   void Show() override;

   UInt_t GetWindowGeometry(Int_t &x, Int_t &y, UInt_t &w, UInt_t &h) override;

   void GetCanvasGeometry(Int_t wid, UInt_t &w, UInt_t &h) override;


   void ShowMenuBar(Bool_t show = kTRUE) override { }
   void ShowStatusBar(Bool_t show = kTRUE) override {  }
   void ShowEditor(Bool_t show = kTRUE) override {  }
   void ShowToolBar(Bool_t show = kTRUE) override { }
   void ShowToolTips(Bool_t show = kTRUE) override {  }

   void   ForceUpdate() override;

   void   SetWindowPosition(Int_t x, Int_t y) override;
   void   SetWindowSize(UInt_t w, UInt_t h) override;
   void   SetWindowTitle(const char *newTitle) override;
   void   SetCanvasSize(UInt_t w, UInt_t h) override;
   void   Iconify() override;
   void   RaiseWindow() override;

   Bool_t HasEditor() const override;
   Bool_t HasMenuBar() const override;
   Bool_t HasStatusBar() const override;
   Bool_t HasToolBar() const override { return kFALSE; }
   Bool_t HasToolTips() const override;

   static TCanvasImp *NewCanvas(TCanvas *c, const char *name, Int_t x, Int_t y, UInt_t width, UInt_t height);

   static TCanvas *CreateQt6Canvas(const char *name, const char *title, UInt_t width = 1200, UInt_t height = 800);

   ClassDefOverride(TQt6Canvas, 0) // Qt6 implementation for TCanvasImp
};

#endif
