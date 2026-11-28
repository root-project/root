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
#include "TQt6Application.h"
#include "TQt6GuiFactory.h"

#include "TSystem.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TROOT.h"
#include "TClass.h"
#include "TError.h"

#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>

#include <QVBoxLayout>

#include "QCanvasWidget.h"

using namespace ROOT::Experimental;

/** \class TQt6Canvas
    \ingroup qt6canvas
    \brief Basic TCanvasImp ABI implementation for Qt6
*/

////////////////////////////////////////////////////////////////////////////////
/// Constructor

TQt6Canvas::TQt6Canvas(TCanvas *c, const char *name, Int_t x, Int_t y, UInt_t width, UInt_t height)
   : TCanvasImp(c, name, x, y, width, height)
{
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
   return new TQt6PadPainter(fPaintWidget);
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Close qt6 canvas - not implemented

void TQt6Canvas::Close()
{
   if (!fCanvasWidget)
      return;

   fCanvasWidget->setAttribute(Qt::WA_DeleteOnClose);
   fCanvasWidget->close();
   fCanvasWidget = nullptr;
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Show qt6 canvas

void TQt6Canvas::Show()
{
   if (fCanvasWidget)
      fCanvasWidget->showNormal();
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

//////////////////////////////////////////////////////////////////////////
/// Set cursor

void TQt6Canvas::SetCursor(ECursor cursor)
{
   if (!fPaintWidget)
      return;

   switch(cursor) {
      case kBottomLeft: fPaintWidget->setCursor(Qt::SizeBDiagCursor); break;
      case kBottomRight: fPaintWidget->setCursor(Qt::SizeFDiagCursor); break;
      case kTopLeft: fPaintWidget->setCursor(Qt::SizeFDiagCursor); break;
      case kTopRight: fPaintWidget->setCursor(Qt::SizeBDiagCursor); break;
      case kBottomSide: fPaintWidget->setCursor(Qt::SizeVerCursor); break;
      case kLeftSide: fPaintWidget->setCursor(Qt::SizeHorCursor); break;
      case kTopSide: fPaintWidget->setCursor(Qt::SizeVerCursor); break;
      case kRightSide: fPaintWidget->setCursor(Qt::SizeHorCursor); break;
      case kMove: fPaintWidget->setCursor(Qt::DragMoveCursor); break;
      case kCross: fPaintWidget->setCursor(Qt::CrossCursor); break;
      case kArrowHor: fPaintWidget->setCursor(Qt::SizeHorCursor); break;
      case kArrowVer: fPaintWidget->setCursor(Qt::UpArrowCursor); break;
      case kHand: fPaintWidget->setCursor(Qt::OpenHandCursor); break;
      case kRotate: fPaintWidget->setCursor(Qt::ClosedHandCursor); break;
      case kPointer: fPaintWidget->setCursor(Qt::ArrowCursor); break;
      case kArrowRight: fPaintWidget->setCursor(Qt::SizeHorCursor); break;
      case kCaret: fPaintWidget->setCursor(Qt::WaitCursor); break;
      case kWatch: fPaintWidget->setCursor(Qt::WaitCursor); break;
      case kNoDrop: fPaintWidget->setCursor(Qt::ForbiddenCursor); break;
      default:
         fPaintWidget->unsetCursor();
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
/// Returns canvas geometry

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

   return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Set visisbility of menu bar

void TQt6Canvas::ShowMenuBar(Bool_t on)
{
   if (fCanvasWidget)
      fCanvasWidget->SetViewMenubar(on);
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Set visisbility of status bar

void TQt6Canvas::ShowStatusBar(Bool_t on)
{
   if (fCanvasWidget)
      fCanvasWidget->SetViewEventStatus(on);
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Set visisbility of tool bar

void TQt6Canvas::ShowToolBar(Bool_t on)
{
   if (fCanvasWidget)
      fCanvasWidget->SetViewToolbar(on);
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Set show flag for tooltips

void TQt6Canvas::ShowToolTips(Bool_t on)
{
   if (fCanvasWidget)
      fCanvasWidget->SetViewToolTip(on);
}

//////////////////////////////////////////////////////////////////////////////////////////
/// if canvas or any subpad was modified,
/// invoke Qt update() which will redraw area

Bool_t TQt6Canvas::PerformUpdate(Bool_t /* async */)
{
   if (Canvas()->IsAnyNeedRepaint() && fPaintWidget)
      fPaintWidget->update();
   return kTRUE;
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Invalide widget area in any case

void TQt6Canvas::ForceUpdate()
{
   if (fPaintWidget)
      fPaintWidget->update();
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/// Static method to create TQt6Canvas instance
/// Used by plugin manager to directly create TQt6Canvas without gui factory

TCanvasImp *TQt6Canvas::NewCanvas(TCanvas *c, const char *name, Int_t x, Int_t y, UInt_t width, UInt_t height)
{
   // ensure QApplication is exists
   TQt6Application::CreateQApplication();

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

   if (imp->fPaintWidget)
      imp->fPaintWidget->SetCanvas(c);

   widget->ApplyCanvasStatusBits();

   // set all internal dimensions
   c->Resize();

   // TODO: maybe apply same logic to adjust canvas dimension as in TRootCanvas
   //       Keep commented code here intentionally to be able find this place when
   //       search for correspondent class members

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
/// Static method to create TCanvas instance with TQt6Canvas implementation and embed
/// it into provided parent widget
/// Can be used by user Qt-based GUIs to embed ROOT graphics

TCanvas *TQt6Canvas::EmbedQt6Canvas(QWidget *parent, const char *name)
{
   if (!parent) {
      ::Error("TQt6Canvas::EmbedQt6Canvas", "parent widget not specified");
      return nullptr;
   }

   // force usage of qt6 gui factory
   if (!dynamic_cast<TQt6GuiFactory *>(gGuiFactory)) {
      delete gGuiFactory;
      gGuiFactory = new TQt6GuiFactory();
   }

   TCanvas *c = new TCanvas(false);
   c->SetName(name);
   c->SetTitle("embed qt6 canvas");
   c->SetBatch(kFALSE);
   // all extra elements are disabled by default
   c->SetBit(TCanvas::kMenuBar, kFALSE);
   c->SetBit(TCanvas::kShowToolBar, kFALSE);
   c->SetBit(TCanvas::kShowEventStatus, kFALSE);

   auto widget = new QCanvasWidget(parent);

   if (parent->layout())
      delete parent->layout();

   QVBoxLayout* layout = new QVBoxLayout(parent);
   layout->setContentsMargins(0, 0, 0, 0);
   layout->addWidget(widget);


   auto imp = new TQt6Canvas(c, name, 0, 0, widget->width(), widget->height());

   imp->fCanvasWidget = widget;
   imp->fPaintWidget = widget->GetPaintWidget();

   if (imp->fPaintWidget)
      imp->fPaintWidget->SetCanvas(c);

   widget->ApplyCanvasStatusBits();

   c->SetCanvasImp(imp);

   return c;
}
