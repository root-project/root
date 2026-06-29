// Author: Sergey Linev, GSI  29/06/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "QPaintWidget.h"

#include "TCanvas.h"

#include <QFont>
#include <QRect>
#include <QPainter>


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
}

QPaintWidget::~QPaintWidget()
{
}

void QPaintWidget::resizeEvent(QResizeEvent *)
{
   printf("Call resize event\n");
   if (fCanvas) {

      fCanvas->Resize();
      fCanvas->Modified();
   }
}

void QPaintWidget::paintEvent(QPaintEvent *)
{
   printf("Call paint event\n");

   try {
      QPainter painter(this);

      fPainter = &painter;

      fCanvas->Paint();

      fPainter = nullptr;

   } catch(...) {
      fPainter = nullptr;
   }
}
