// Author: Sergey Linev, GSI  13/01/2021

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "RCanvasWidget.h"

#include <ROOT/RCanvas.hxx>

#include <QGridLayout>

RCanvasWidget::RCanvasWidget(QWidget *parent) : QWidget(parent)
{

   setObjectName( "RCanvasWidget");

   setSizeIncrement( QSize( 100, 100 ) );

   setUpdatesEnabled( true );
   setMouseTracking(true);

   setFocusPolicy( Qt::TabFocus );
   setCursor( Qt::CrossCursor );

   setAcceptDrops(true);

   QGridLayout *gridLayout = new QGridLayout(this);
   gridLayout->setSpacing(10);
   gridLayout->setMargin(1);

   fCanvas = new ROOT::Experimental::RCanvas();

   std::string arg = "qt5:";
   arg += std::to_string((unsigned long) this);
   arg += "?noopenui";

   fCanvas->Show(arg);

   fView = findChild<QWebEngineView*>("RootWebView");
   if (!fView) {
      printf("FAIL TO FIND QWebEngineView - ROOT Qt5Web plugin does not work properly !!!!!\n");
      exit(11);
   }

   gridLayout->addWidget(fView);

   fCanvas->SetSize({fView->width(), fView->height()});
}

RCanvasWidget::~RCanvasWidget()
{
   if (fCanvas) {
      delete fCanvas;
      fCanvas = nullptr;
   }
}

void RCanvasWidget::resizeEvent(QResizeEvent *event)
{
   fCanvas->SetSize({fView->width(), fView->height()});
}
