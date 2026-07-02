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
#include "TError.h"
#include "TCanvas.h"
#include "TThread.h"
#include "TROOT.h"
#include "TApplication.h"
#include "TVirtualX.h"
#include "TVirtualPS.h"
#include "TClass.h"
#include "TExec.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include <QApplication>
#include "QCanvasWidget.h"

class TQt6CanvasTimer : public TTimer {
public:
   TQt6CanvasTimer(Long_t milliSec, Bool_t mode) :
       TTimer(milliSec, mode) {}


   /// used to send control messages to clients
   void Timeout() override
   {
      QApplication::sendPostedEvents();
      QApplication::processEvents();
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
   TThread::SelfId();

   // fTimer = new TQt6CanvasTimer(*this);

   // fTimer->TurnOn();

   // fAsyncMode = kTRUE;
}


////////////////////////////////////////////////////////////////////////////////
/// Destructor

TQt6Canvas::~TQt6Canvas()
{
   // delete fTimer;
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
   printf("Create pad painter\n");
   return new TQt6PadPainter(fPaintWidget);
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
   if (fCanvasWidget)
      fCanvasWidget->move(x, y);
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Set window size of web canvas

void TQt6Canvas::SetWindowSize(UInt_t w, UInt_t h)
{
   if (fCanvasWidget)
      fCanvasWidget->resize(w, h);
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Set window title of web canvas

void TQt6Canvas::SetWindowTitle(const char *newTitle)
{
   if (fCanvasWidget)
      fCanvasWidget->setWindowTitle(QString::fromLatin1(newTitle));
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Set canvas size

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
   if (fCanvasWidget)
      fCanvasWidget->showMinimized();
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Raise browser window

void TQt6Canvas::RaiseWindow()
{
   if (fCanvasWidget) {
      fCanvasWidget->showNormal();
      fCanvasWidget->raise();
      fCanvasWidget->activateWindow();
   }
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

//////////////////////////////////////////////////////////////////////////////////////////////////
/// Process TExec objects in the pad

void TQt6Canvas::ProcessExecs(TPad *pad, TExec *extra)
{
   auto execs = pad ? pad->GetListOfExecs() : nullptr;

   if ((!execs || !execs->GetSize()) && !extra)
      return;

   auto saveps = gVirtualPS;
   gVirtualPS = nullptr;

   auto savex = gVirtualX;
   TVirtualX x;
   gVirtualX = &x;

   TIter next(execs);
   while (auto obj = next()) {
      auto exec = dynamic_cast<TExec *>(obj);
      if (exec)
         exec->Exec();
   }

   if (extra)
      extra->Exec();

   gVirtualPS = saveps;
   gVirtualX = savex;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/// Return canvas geometry

void TQt6Canvas::GetCanvasGeometry(Int_t wid, UInt_t &w, UInt_t &h)
{
   (void) wid;
   if (fPaintWidget) {
      w = fPaintWidget->width();
      h = fPaintWidget->height();
   } else {
      w = 780;
      h = 580;
   }
}


//////////////////////////////////////////////////////////////////////////////////////////
/// Returns window geometry including borders and menus

UInt_t TQt6Canvas::GetWindowGeometry(Int_t &x, Int_t &y, UInt_t &w, UInt_t &h)
{
   if (fCanvasWidget) {
      auto pos = fCanvasWidget->pos();
      x = pos.x();
      y = pos.y();
      w = fCanvasWidget->width();
      h = fCanvasWidget->height();
   } else {
      x = y = 0;
      w = 800;
      h = 600;
   }

   // x = Canvas()->fWindowTopX;
   // y = Canvas()->fWindowTopY;
   // w = Canvas()->fWindowWidth;
   // h = Canvas()->fWindowHeight;

   return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////
/// if canvas or any subpad was modified,
/// invoke Qt update() which will redraw area

Bool_t TQt6Canvas::PerformUpdate(Bool_t /* async */)
{
   if (Canvas()->IsModified())
      fPaintWidget->update();
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
   static QApplication *qapp = nullptr;
   static int qargc = 1;
   static char *qargv[2];

   if (!qapp && !QApplication::instance()) {

      if (!gApplication) {
         ::Error("TQt6Canvas::NewCanvas", "Not found gApplication to create QApplication");
         return nullptr;
      }

      qargv[0] = gApplication->Argv(0);
      qargv[1] = nullptr;

      qapp = new QApplication(qargc, qargv);
   }

   static TQt6CanvasTimer *timer = nullptr;

   if (!timer) {
      timer = new TQt6CanvasTimer(10, kTRUE);
      timer->TurnOn();
   }

   auto widget = new QCanvasWidget();
   widget->setWindowTitle(QString(c->GetTitle()));
   if ((x < 0) && (y < 0))
      widget->resize(width, height);
   else
      widget->setGeometry(x, y, width, height);
   widget->show();

   auto imp = new TQt6Canvas(c, name, x, y, width, height);

   imp->fCanvasWidget = widget;
   imp->fPaintWidget = widget->GetPaintWidget();

   imp->fPaintWidget->SetCanvas(c);

   // set all internal dimensions
   c->Resize();

   // c->fWindowTopX = x;
   // c->fWindowTopY = y;
   // c->fWindowWidth = width;
   // c->fWindowHeight = height;
   // if (!gROOT->IsBatch() && (height > 25))
   //   height -= 25;
   // c->fCw = width;
   // c->fCh = height;

   return imp;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/// Create TCanvas and assign TQt6Canvas implementation to it
/// Canvas is not displayed automatically, therefore canv->Show() method must be called
/// Or canvas can be embed in other widgets.

TCanvas *TQt6Canvas::CreateQt6Canvas(const char *name, const char *title, UInt_t width, UInt_t height)
{
   auto canvas = new TCanvas(kFALSE);
   canvas->SetName(name);
   canvas->SetTitle(title);
   canvas->ResetBit(TCanvas::kShowEditor);
   canvas->ResetBit(TCanvas::kShowToolBar);
   canvas->SetBit(TCanvas::kMenuBar, kTRUE);
   canvas->SetCanvas(canvas);
   canvas->SetBatch(kTRUE); // mark canvas as batch
   canvas->SetEditable(kTRUE); // ensure fPrimitives are created

   // copy gStyle attributes
   canvas->SetFillColor(gStyle->GetCanvasColor());
   canvas->SetFillStyle(1001);
   canvas->SetGrid(gStyle->GetPadGridX(),gStyle->GetPadGridY());
   canvas->SetTicks(gStyle->GetPadTickX(),gStyle->GetPadTickY());
   canvas->SetLogx(gStyle->GetOptLogx());
   canvas->SetLogy(gStyle->GetOptLogy());
   canvas->SetLogz(gStyle->GetOptLogz());
   canvas->SetBottomMargin(gStyle->GetPadBottomMargin());
   canvas->SetTopMargin(gStyle->GetPadTopMargin());
   canvas->SetLeftMargin(gStyle->GetPadLeftMargin());
   canvas->SetRightMargin(gStyle->GetPadRightMargin());
   canvas->SetBorderSize(gStyle->GetCanvasBorderSize());
   canvas->SetBorderMode(gStyle->GetCanvasBorderMode());

   auto imp = static_cast<TQt6Canvas *> (NewCanvas(canvas, name, 0, 0, width, height));

   canvas->SetCanvasImp(imp);

   canvas->cd();

   {
      R__LOCKGUARD(gROOTMutex);
      auto l1 = gROOT->GetListOfCleanups();
      if (!l1->FindObject(canvas))
         l1->Add(canvas);
      auto l2 = gROOT->GetListOfCanvases();
      if (!l2->FindObject(canvas))
         l2->Add(canvas);
   }

   return canvas;
}
