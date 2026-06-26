// Author: Sergey Linev, GSI   26/06/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TQt6Canvas.h"

#include "TQt6PadPainter.h"

#include "TSystem.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TThread.h"
#include "TROOT.h"
#include "TClass.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

using namespace ROOT::Experimental;

class TQt6CanvasTimer : public TTimer {
   TQt6Canvas &fCanv;
public:
   TQt6CanvasTimer(TQt6Canvas &canv) : TTimer(10, kTRUE), fCanv(canv) {}


   /// used to send control messages to clients
   void Timeout() override
   {
   }
};


/** \class TQt6Canvas
    \ingroup qt6canvas
    \brief Basic TCanvasImp ABI implementation for Qt6

*/

using namespace std::string_literals;

////////////////////////////////////////////////////////////////////////////////
/// Constructor

TQt6Canvas::TQt6Canvas(TCanvas *c, const char *name, Int_t x, Int_t y, UInt_t width, UInt_t height)
   : TCanvasImp(c, name, x, y, width, height)
{
   // Workaround for multi-threaded environment
   // Ensure main thread id picked when canvas implementation is created -
   // otherwise it may be assigned in other thread and screw-up gPad access.
   // Workaround may not work if main thread id was wrongly initialized before
   // This resolves issue https://github.com/root-project/root/issues/15498
   // TThread::SelfId();

   fTimer = new TQt6CanvasTimer(*this);

   fTimer->TurnOn();

   // fAsyncMode = kTRUE;
}


////////////////////////////////////////////////////////////////////////////////
/// Destructor

TQt6Canvas::~TQt6Canvas()
{
   delete fTimer;
}


////////////////////////////////////////////////////////////////////////////////
/// Initialize window for the qt6 canvas

Int_t TQt6Canvas::InitWindow()
{
   return 111222333; // should not be used at all
}

////////////////////////////////////////////////////////////////////////////////
/// Creates pad painter

TVirtualPadPainter *TQt6Canvas::CreatePadPainter()
{
   return new TQt6PadPainter();
}


//////////////////////////////////////////////////////////////////////////////////////////
/// Close qt6 canvas - not implemented

void TQt6Canvas::Close()
{
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Show qt6 canvas

void TQt6Canvas::Show()
{
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Returns kTRUE if web canvas has graphical editor

Bool_t TQt6Canvas::HasEditor() const
{
   return (fClientBits & TCanvas::kShowEditor) != 0;
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Returns kTRUE if web canvas has menu bar

Bool_t TQt6Canvas::HasMenuBar() const
{
   return (fClientBits & TCanvas::kMenuBar) != 0;
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Returns kTRUE if web canvas has status bar

Bool_t TQt6Canvas::HasStatusBar() const
{
   return (fClientBits & TCanvas::kShowEventStatus) != 0;
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Returns kTRUE if tooltips are activated in web canvas

Bool_t TQt6Canvas::HasToolTips() const
{
   return (fClientBits & TCanvas::kShowToolTips) != 0;
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Set window position of web canvas

void TQt6Canvas::SetWindowPosition(Int_t x, Int_t y)
{
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Set window size of web canvas

void TQt6Canvas::SetWindowSize(UInt_t w, UInt_t h)
{
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Set window title of web canvas

void TQt6Canvas::SetWindowTitle(const char *newTitle)
{
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Set canvas size of web canvas

void TQt6Canvas::SetCanvasSize(UInt_t cw, UInt_t ch)
{
   fFixedSize = kTRUE;
   if ((cw > 0) && (ch > 0)) {
      // Canvas()->fCw = cw;
      // Canvas()->fCh = ch;
   } else {
      // temporary value, will be reported back from client
      // Canvas()->fCw = Canvas()->fWindowWidth;
      // Canvas()->fCh = Canvas()->fWindowHeight;
   }
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Iconify browser window

void TQt6Canvas::Iconify()
{
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Raise browser window

void TQt6Canvas::RaiseWindow()
{
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Assign clients bits

void TQt6Canvas::AssignStatusBits(UInt_t bits)
{
   fClientBits = bits;
   Canvas()->SetBit(TCanvas::kShowEventStatus, bits & TCanvas::kShowEventStatus);
   Canvas()->SetBit(TCanvas::kShowEditor, bits & TCanvas::kShowEditor);
   Canvas()->SetBit(TCanvas::kShowToolTips, bits & TCanvas::kShowToolTips);
   Canvas()->SetBit(TCanvas::kMenuBar, bits & TCanvas::kMenuBar);
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Returns window geometry including borders and menus

UInt_t TQt6Canvas::GetWindowGeometry(Int_t &x, Int_t &y, UInt_t &w, UInt_t &h)
{
   // x = Canvas()->fWindowTopX;
   // y = Canvas()->fWindowTopY;
   // w = Canvas()->fWindowWidth;
   // h = Canvas()->fWindowHeight;

   return 0;
}


//////////////////////////////////////////////////////////////////////////////////////////
/// if canvas or any subpad was modified,
/// scan all primitives in the TCanvas and subpads and convert them into
/// the structure which will be delivered to JSROOT client

Bool_t TQt6Canvas::PerformUpdate(Bool_t async)
{
   return kTRUE;
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Increment canvas version and force sending data to client - do not wait for reply

void TQt6Canvas::ForceUpdate()
{
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/// Static method to create TQt6Canvas instance
/// Used by plugin manager

TCanvasImp *TQt6Canvas::NewCanvas(TCanvas *c, const char *name, Int_t x, Int_t y, UInt_t width, UInt_t height)
{
   auto imp = new TQt6Canvas(c, name, x, y, width, height);

   // c->fWindowTopX = x;
   // c->fWindowTopY = y;
   // c->fWindowWidth = width;
   // c->fWindowHeight = height;
   if (!gROOT->IsBatch() && (height > 25))
      height -= 25;
   // c->fCw = width;
   // c->fCh = height;

   return imp;
}
