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

#include <QtCore/QTimer>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QMouseEvent>
#include <QWheelEvent>
#include <QHelpEvent>
#include <QCloseEvent>

#include <QToolTip>
#include <QFont>
#include <QRect>
#include <QPainter>

/** \class QPaintWidget
    \ingroup qt6canvas

Represent area where actual canvas painting is performed

Qt defines `paintEvent` virtual method where all painting should be done.
At this moment QPainter object is created which then used in \ref TVirtualPadPainter API.
*/


////////////////////////////////////////////////////////////////////////////////
/// Constructor

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

////////////////////////////////////////////////////////////////////////////////
/// Destructor

QPaintWidget::~QPaintWidget()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Convert mouse event position to canvas coordinates

QPoint QPaintWidget::scaledMousePoint(QMouseEvent *e)
{
   int scaledX = scaledPosition(e->position().x());
   int scaledY = scaledPosition(e->position().y());
   return QPoint(scaledX, scaledY);
}

////////////////////////////////////////////////////////////////////////////////
/// Generic event handler
///
/// Used to handle tooltip event - if configured

bool QPaintWidget::event(QEvent *event)
{
   if (fShowToolTip && (event->type() == QEvent::ToolTip)) {
      auto *helpEvent = static_cast<QHelpEvent *>(event);

      TObject *selected = fCanvas->GetSelected();
      Int_t px = fCanvas->GetEventX();
      Int_t py = fCanvas->GetEventY();
      QString customText;
      if (selected) {
         customText = QString("%1::%2<br>%3<br>%4, %5<br>%6")
                       .arg(selected->ClassName()).arg(selected->GetName())
                       .arg(selected->GetTitle())
                       .arg(px).arg(py)
                       .arg(selected->GetObjectInfo(px, py));
      } else {
         customText = QString("No selected object<br>%1, %2")
                       .arg(px).arg(py);
      }

      // 4. Force screen coordinates to bypass standard widget offset logic
      QToolTip::showText(helpEvent->globalPos(), customText, this);

      return true; // Event handled successfully
   }

   return QWidget::event(event);
}

////////////////////////////////////////////////////////////////////////////////
/// Resize event handler

void QPaintWidget::resizeEvent(QResizeEvent *)
{
   if (fCanvas) {
      fCanvas->Resize();
      fCanvas->Modified();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Paint event handler
///
/// Only inside this method painting on QWidget is allowed
/// For this temporary QPainter object is created and assigned to fPainter
/// Any attempt to paint outside this method will be blocked - while fPainter will be 0


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

////////////////////////////////////////////////////////////////////////////////
/// Mouse press event handler

void QPaintWidget::mousePressEvent(QMouseEvent *e)
{
   QPoint scaled = scaledMousePoint(e);

   switch(e->button()) {
     case Qt::LeftButton:
        fCanvas->HandleInput(kButton1Down, scaled.x(), scaled.y());
        // emit PadClicked(pad, scaled.x(), scaled.y());
        break;
     case Qt::RightButton : {
        fCanvas->HandleInput(kButton3Down, scaled.x(), scaled.y());
        break;
     }
     case Qt::MiddleButton :
        fCanvas->HandleInput(kButton2Down, scaled.x(), scaled.y());
        // emit SelectedPadChanged(pad);
        break;
     case Qt::NoButton :
        break;
     default:
        break;
   }
   e->accept();
}


////////////////////////////////////////////////////////////////////////////////
/// Mouse move event handler

void QPaintWidget::mouseMoveEvent(QMouseEvent *e)
{
   e->accept();

   if (fCanvas) {
      QPoint pnt = scaledMousePoint(e);

      if (e->buttons() & Qt::LeftButton)
        fCanvas->HandleInput(kButton1Motion, pnt.x(), pnt.y());
      else
        fCanvas->HandleInput(kMouseMotion, pnt.x(), pnt.y());
   }

   if (fShowEventStatus) {
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

      emit CanvasStatusEvent(buffer);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Mouse release event handler

void QPaintWidget::mouseReleaseEvent(QMouseEvent *event)
{
   QPoint scaled = scaledMousePoint(event);

   switch(event->button()) {
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

   event->accept();
}

////////////////////////////////////////////////////////////////////////////////
/// Mouse double click event handler

void QPaintWidget::mouseDoubleClickEvent(QMouseEvent *event)
{
   QPoint scaled = scaledMousePoint(event);

   switch(event->button()) {
      case Qt::LeftButton : {
         if (!fMaskDoubleClick)
            fCanvas->HandleInput(kButton1Double, scaled.x(), scaled.y());
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

   event->accept();
}

////////////////////////////////////////////////////////////////////////////////
/// Wheel event handler

void QPaintWidget::wheelEvent(QWheelEvent *event)
{
   QPoint delta = event->pixelDelta();
   if (delta.isNull())
      delta = event->angleDelta() / 8;
   bool positive = delta.x() > 0 || delta.y() > 0;

   int sx = scaledPosition(event->position().x());
   int sy = scaledPosition(event->position().y());

   fCanvas->HandleInput(positive ? kWheelUp : kWheelDown, sx, sy);

   event->accept();
}

////////////////////////////////////////////////////////////////////////////////
/// Enter event handler

void QPaintWidget::enterEvent(QEnterEvent *event)
{
   QWidget::enterEvent(event);

   fCanvas->HandleInput(kMouseEnter, 0, 0);
}

////////////////////////////////////////////////////////////////////////////////
/// Leave event handler

void QPaintWidget::leaveEvent(QEvent *event)
{
   QWidget::leaveEvent(event);

   fCanvas->HandleInput(kMouseLeave, 0, 0);
}
