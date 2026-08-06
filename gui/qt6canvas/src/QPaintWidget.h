// Author: Sergey Linev, GSI  29/06/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_QPaintWidget
#define ROOT_QPaintWidget

#include <QWidget>

class TCanvas;
class TPad;
class TMethod;
class QPainter;

class QPaintWidget : public QWidget {

   Q_OBJECT

public:
   QPaintWidget(QWidget *parent = nullptr);
   virtual ~QPaintWidget();

   /// returns canvas shown in the widget
   TCanvas *getCanvas() { return fCanvas; }

   QPainter *getPainter() const { return fPainter; }

   void SetCanvas(TCanvas *canv) { fCanvas = canv; }

   void SetShowToolTip(bool on) { fShowToolTip = on; }

signals:
   void CanvasStatusEvent(const QString &msg);

protected:
   bool event(QEvent *event) override;

   void resizeEvent(QResizeEvent *event) override;

   void paintEvent(QPaintEvent *event) override;

   void mousePressEvent(QMouseEvent *event) override;
   void mouseMoveEvent(QMouseEvent *event) override;
   void mouseReleaseEvent(QMouseEvent *event) override;
   void mouseDoubleClickEvent(QMouseEvent* event) override;
   void wheelEvent(QWheelEvent* event) override;
   void enterEvent(QEnterEvent *event) override;
   void leaveEvent(QEvent *event) override;

   double scaledPosition(int p) { return (double) p * fQtScalingfactor; }

   QPoint scaledMousePoint(QMouseEvent *event);

   TCanvas *fCanvas = nullptr; ///<  canvas shown in the widget

   QPainter *fPainter = nullptr; ///< temporary painter created inside paintEvent handler

   double fQtScalingfactor = 1.;

   bool              fMaskDoubleClick = false;
   bool              fShowEventStatus = true;
   bool              fShowToolTip = false;
};

#endif
