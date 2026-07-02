// Author: Sergey Linev, GSI  29/06/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef QPaintWidget_H
#define QPaintWidget_H

#include <QWidget>

class TCanvas;
class TPad;
class TMethod;
class QPainter;
class QStatusBar;

class QPaintWidget : public QWidget {

   Q_OBJECT

public:
   QPaintWidget(QWidget *parent = nullptr);
   virtual ~QPaintWidget();

   /// returns canvas shown in the widget
   TCanvas *getCanvas() { return fCanvas; }

   QPainter *getPainter() const { return fPainter; }

   void SetCanvas(TCanvas *canv) { fCanvas = canv; }

   void SetStatusBar(QStatusBar *bar) { fStatusBar = bar; }

protected:
   void resizeEvent(QResizeEvent *event) override;

   void paintEvent(QPaintEvent *event) override;

   void mousePressEvent(QMouseEvent *event) override;
   void mouseMoveEvent(QMouseEvent *event) override;
   void mouseReleaseEvent(QMouseEvent *event) override;
   void mouseDoubleClickEvent(QMouseEvent* event) override;

   double scaledPosition(int p) { return (double) p * fQtScalingfactor; }

   QPoint scaledMousePoint(QMouseEvent *event);

   TCanvas *fCanvas = nullptr;

   QPainter *fPainter = nullptr;

   double fQtScalingfactor = 1.;

   QStatusBar  *fStatusBar = nullptr;

   bool              fMaskDoubleClick = false;
   bool              fShowEventStatus = true;

};

#endif
