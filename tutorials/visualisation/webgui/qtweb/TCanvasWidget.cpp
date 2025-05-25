// Author: Sergey Linev, GSI  13/01/2021

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TCanvasWidget.h"

#include "TCanvas.h"
#include "TROOT.h"
#include "TClass.h"
#include "TEnv.h"

#include "TWebCanvas.h"

TCanvasWidget::TCanvasWidget(QWidget *parent) : QWidget(parent)
{
   setObjectName("TCanvasWidget");

   setSizeIncrement(QSize(100, 100));

   setUpdatesEnabled(true);
   setMouseTracking(true);

   setFocusPolicy(Qt::TabFocus);
   setCursor(Qt::CrossCursor);

   setAcceptDrops(true);

   static int wincnt = 1;

   auto name = TString::Format("Canvas%d", wincnt++);

   fCanvas = TWebCanvas::CreateWebCanvas(name, name);

   auto web = static_cast<TWebCanvas *> (fCanvas->GetCanvasImp());

   web->SetCanCreateObjects(kFALSE); // not yet create objects on server side

   web->SetUpdatedHandler([this]() { emit CanvasUpdated(); });

   web->SetActivePadChangedHandler([this](TPad *pad) { emit SelectedPadChanged(pad); });

   web->SetPadClickedHandler([this](TPad *pad, int x, int y) { emit PadClicked(pad, x, y); });

   web->SetPadDblClickedHandler([this](TPad *pad, int x, int y) { emit PadDblClicked(pad, x, y); });

   auto where = ROOT::RWebDisplayArgs::GetQtEmbedQualifier(this, "noopenui", QT_VERSION);

   web->ShowWebWindow(where);

   fView = findChild<QWebEngineView *>("RootWebView");
   if (!fView) {
      printf("FAIL TO FIND QWebEngineView - ROOT Qt5Web plugin does not work properly !!!!!\n");
      exit(11);
   }

   fView->resize(width(), height());
   fCanvas->SetCanvasSize(width(), height());
}

TCanvasWidget::~TCanvasWidget()
{
   if (fCanvas) {
      gROOT->GetListOfCanvases()->Remove(fCanvas);

      fCanvas->SetCanvasImp(nullptr);

      fCanvas->Close();
      delete fCanvas;
      fCanvas = nullptr;
   }
}

void TCanvasWidget::resizeEvent(QResizeEvent *event)
{
   fView->resize(width(), height());
   fCanvas->SetCanvasSize(width(), height());
}

void TCanvasWidget::activateEditor(TPad *pad, TObject *obj)
{
   TWebCanvas *cimp = dynamic_cast<TWebCanvas *>(fCanvas->GetCanvasImp());
   if (cimp) {
      cimp->ShowEditor(kTRUE);
      cimp->ActivateInEditor(pad, obj);
   }
}

void TCanvasWidget::setEditorVisible(bool flag)
{
   TCanvasImp *cimp = fCanvas->GetCanvasImp();
   if (cimp)
      cimp->ShowEditor(flag);
}

void TCanvasWidget::activateStatusLine()
{
   TCanvasImp *cimp = fCanvas->GetCanvasImp();
   if (cimp)
      cimp->ShowStatusBar(kTRUE);
}
