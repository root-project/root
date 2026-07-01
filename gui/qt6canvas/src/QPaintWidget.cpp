// Author: Sergey Linev, GSI  29/06/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "QPaintWidget.h"

#include <iostream>

#include "TCanvas.h"
#include "TROOT.h"
#include "TMethod.h"

#include "QRootMethodDialog.h"

#include <QtCore/QSignalMapper>
#include <QtCore/QTimer>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QMouseEvent>
#include <QCloseEvent>
#include <QMenu>
#include <QAction>
#include <QFont>
#include <QRect>
#include <QPainter>
#include <QStatusBar>


QPaintWidget::QPaintWidget(QWidget *parent) : QWidget(parent)
{
   setObjectName("QPaintWidget");

   setSizeIncrement(QSize(100, 100));

   setUpdatesEnabled(true);
   setMouseTracking(true);

   setFocusPolicy(Qt::TabFocus);
   setCursor(Qt::CrossCursor);

   setAcceptDrops(true);

   fCanvas = nullptr;

   fQtScalingfactor = (double) metric(QPaintDevice::PdmDevicePixelRatioScaled)/65536.;
}

QPaintWidget::~QPaintWidget()
{
}

QPoint QPaintWidget::scaledMousePoint(QMouseEvent *e)
{
   int scaledX = scaledPosition(e->position().x());
   int scaledY = scaledPosition(e->position().y());
   return QPoint(scaledX, scaledY);
}


void QPaintWidget::resizeEvent(QResizeEvent *)
{
   if (fCanvas) {
      fCanvas->Resize();
      fCanvas->Modified();
   }
}

void QPaintWidget::paintEvent(QPaintEvent *)
{
   try {
      QPainter painter(this);

      fPainter = &painter;

      fCanvas->Paint();

      fPainter = nullptr;

   } catch(...) {
      fPainter = nullptr;
   }
}

void QPaintWidget::mousePressEvent( QMouseEvent *e )
{
   TObjLink* pickobj = nullptr;
   QPoint scaled = scaledMousePoint(e);
   QPoint menu_pnt = e->globalPosition().toPoint();

   TPad *pad = fCanvas->Pick(scaled.x(), scaled.y(), pickobj);
   TObject *selected = fCanvas->GetSelected();

   switch(e->button()) {
     case Qt::LeftButton :
        fCanvas->HandleInput(kButton1Down, scaled.x(), scaled.y());
        // emit PadClicked(pad, scaled.x(), scaled.y());
        break;
     case Qt::RightButton : {
        TString selectedOpt;
        if (pad) {
           if (!pickobj) {
              fCanvas->SetSelected(pad);
              selected = pad;
           } else if(!selected) {
              selected    = pickobj->GetObject();
              selectedOpt = pickobj->GetOption();
           }
           pad->cd();
        }
        fCanvas->SetSelectedPad(pad);
        gROOT->SetSelectedPrimitive(selected);
        fMousePosX = gPad->AbsPixeltoX(gPad->GetEventX());
        fMousePosY = gPad->AbsPixeltoY(gPad->GetEventY());

        QMenu menu;
        QSignalMapper map;

        QObject::connect(&map, &QSignalMapper::mappedInt,
                         this, &QPaintWidget::executeMenu);
        fMenuObj = selected;
        fMenuMethods = new TList;
        TClass *cl = fMenuObj->IsA();
        int curId = -1;

        QString buffer = TString::Format("%s::%s", cl->GetName(), fMenuObj->GetName()).Data();
        addMenuAction(&menu, &map, buffer, curId++);

        cl->GetMenuItems(fMenuMethods);
        menu.addSeparator();

        TIter iter(fMenuMethods);
        while (auto method = dynamic_cast<TMethod*>(iter())) {
           buffer = method->GetName();
           addMenuAction(&menu, &map, buffer, curId++);
        }

        if (menu.exec(menu_pnt) == nullptr) {
           fMenuObj = nullptr;
           delete fMenuMethods;
           fMenuMethods = nullptr;
        }

        break;
     }
     case Qt::MiddleButton :
        fCanvas->HandleInput(kButton2Down, scaled.x(), scaled.y());
        // emit SelectedPadChanged(pad);
        break;
     case  Qt::NoButton :
        break;
     default:
        break;
   }
   e->accept();
}


void QPaintWidget::mouseMoveEvent(QMouseEvent *e)
{
   static ulong lastprocesstime = 0;
   static ulong delta = 100;
   ulong timestamp = e->timestamp();
   e->accept();
   if(timestamp - delta < lastprocesstime)
      return;
   lastprocesstime = timestamp;

   if (fCanvas) {
      QPoint pnt = scaledMousePoint(e);

      if (e->buttons() & Qt::LeftButton)
        fCanvas->HandleInput(kButton1Motion, pnt.x(), pnt.y());
      else
        fCanvas->HandleInput(kMouseMotion, pnt.x(), pnt.y());
   }

  if(fShowEventStatus) {
     TObject *selected = fCanvas->GetSelected();
     Int_t px = fCanvas->GetEventX();
     Int_t py = fCanvas->GetEventY();
     QString buffer = "";
     if (selected) {
        buffer = selected->GetName();
        buffer += "  ";
        buffer += selected->GetObjectInfo(px, py);
     } else {
        buffer = "No selected object x = ";
        buffer += QString::number(px);
        buffer += "  y = ";
        buffer += QString::number(py);
     }

     // emit CanvasStatusEvent(buffer.toLatin1().constData());

     if (fStatusBar) fStatusBar->showMessage(buffer.toLatin1().constData());
  }
}

void QPaintWidget::mouseReleaseEvent( QMouseEvent *e )
{
   QPoint scaled = scaledMousePoint(e);

   switch(e->button()) {
      case Qt::LeftButton :
         fCanvas->HandleInput(kButton1Up, scaled.x(), scaled.y());
         break;
      case Qt::RightButton :
         fCanvas->HandleInput(kButton3Up, scaled.x(), scaled.y());
         break;
      case Qt::MiddleButton :
         fCanvas->HandleInput(kButton2Up, scaled.x(), scaled.y());
         break;
      case Qt::NoButton :
         break;
      default:
         break;
   }
   e->accept();
}

void QPaintWidget::mouseDoubleClickEvent( QMouseEvent *e )
{
   QPoint scaled = scaledMousePoint(e);

   switch(e->button()) {
      case Qt::LeftButton : {
         if (!fMaskDoubleClick)
            fCanvas->HandleInput(kButton1Double, scaled.x(), scaled.y());
         TObjLink* pickobj = nullptr;
         TPad *pad = fCanvas->Pick(scaled.x(), scaled.y(), pickobj);
         // emit PadDoubleClicked(pad, scaled.x(), scaled.y());
         break;
      }
      case Qt::RightButton :
         fCanvas->HandleInput(kButton3Double, scaled.x(), scaled.y());
         break;
      case Qt::MiddleButton :
         fCanvas->HandleInput(kButton2Double, scaled.x(), scaled.y());
         break;
      case Qt::NoButton :
         break;
      default:
         break;
   }
    e->accept();
}

QAction* QPaintWidget::addMenuAction(QMenu* menu, QSignalMapper* map, const QString& text, int id)
{
   bool enabled = true;

   QAction* act = new QAction(text, menu);

   if (!enabled)
      if ((text.compare("DrawClone") == 0) || (text.compare("DrawClass") == 0) || (text.compare("Inspect") == 0) ||
          (text.compare("SetShowProjectionX") == 0) || (text.compare("SetShowProjectionY") == 0) ||
          (text.compare("DrawPanel") == 0) || (text.compare("FitPanel") == 0))
         act->setEnabled(false);

   QObject::connect(act, &QAction::triggered, [id, map]() {
      map->mappedInt(id);
   });

   menu->addAction(act);
   map->setMapping(act, id);

   return act;
}

void QPaintWidget::executeMenu(int id)
{
   QString text;
   bool ok = false;
   if (id >= 0) {

      // save global to Pad before calling TObject::Execute()

      TVirtualPad *psave = gROOT->GetSelectedPad();
      TMethod *method = (TMethod *) fMenuMethods->At(id);

      /// test: do this in any case!
      fCanvas->HandleInput(kButton3Up, gPad->XtoAbsPixel(fMousePosX), gPad->YtoAbsPixel(fMousePosY));

      // change current dir that all new histograms appear here
      gROOT->cd();

      if (method->GetListOfMethodArgs()->First()) {
         QRootMethodDialog dlg;

         dlg.methodDialog(fMenuObj, method);
      } else {
         gROOT->SetFromPopUp(kTRUE);
         fMenuObj->Execute(method->GetName(), "");

         if (fMenuObj->TestBit(TObject::kNotDeleted)) {
            // emit MenuCommandExecuted(fMenuObj, method->GetName());
         } else {
            fMenuObj = nullptr;
         }

      }

      fCanvas->GetPadSave()->Update();
      fCanvas->GetPadSave()->Modified();

      gROOT->SetSelectedPad(psave);

      gROOT->GetSelectedPad()->Update();
      gROOT->GetSelectedPad()->Modified();

      fCanvas->Modified();
      fCanvas->ForceUpdate();
      gROOT->SetFromPopUp(kFALSE);
    }

   fMenuObj = nullptr;
   delete fMenuMethods;
   fMenuMethods = nullptr;
}
