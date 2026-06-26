// Author: Sergey Linev, GSI  26/06/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "QCanvasWidget.h"

#include "TCanvas.h"
#include "TROOT.h"

#include <QMessageBox>
#include <QPushButton>

QCanvasWidget::QCanvasWidget(QWidget *parent, const char *name) : QWidget(parent)
{
   setupUi(this);

   setAttribute(Qt::WA_DeleteOnClose);

   setObjectName(name);


   fMenuBar = new QMenuBar(fMenuFrame);
   fMenuBar->setMinimumWidth(50);
   fMenuBar->setNativeMenuBar(kFALSE); // disable putting this to screen menu. for MAC style WMs

   QMenu* fileMenu = fMenuBar->addMenu("F&ile");
   fileMenu->addAction("&New canvas", this, &QCanvasWidget::NewCanvas);
   fileMenu->addAction("Open ...", this, &QCanvasWidget::OpenRootFile);
   fileMenu->addAction("Cl&ose canvas", this, &QCanvasWidget::CloseCanvas);

   fileMenu->addSeparator();

   fileMenu->addAction("Save as...", this, &QCanvasWidget::SaveCanvasAs);

   fileMenu->addSeparator();

   fileMenu->addAction("Print...", this, &QCanvasWidget::PrintCanvas);
   fileMenu->addSeparator();

   fileMenu->addAction("Quit ROOT", this, &QCanvasWidget::QuitRoot);


   fMenuBar->addMenu("&Edit");

   fMenuBar->addMenu("&View");

   fMenuBar->addMenu("&Options");

   fMenuBar->addMenu("&Tools");

   QWidget *spacer = new QWidget(this);
   spacer->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
   fMenuBar->setCornerWidget(spacer, Qt::TopRightCorner);

   fMenuBar->addMenu("&Help");

}

QCanvasWidget::~QCanvasWidget() {}


void QCanvasWidget::NewCanvas()
{

}

void QCanvasWidget::OpenRootFile()
{

}

void QCanvasWidget::CloseCanvas()
{
   close();
}

void QCanvasWidget::SaveCanvasAs()
{

}

void QCanvasWidget::PrintCanvas()
{

}

void QCanvasWidget::QuitRoot()
{
}


