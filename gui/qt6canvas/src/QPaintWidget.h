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
class TObject;
class TList;
class TMethod;
class QPainter;
class QSignalMapper;
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

public slots:
   void executeMenu(int id);

protected:
   void resizeEvent(QResizeEvent *event) override;

   void paintEvent(QPaintEvent *event) override;

   void mousePressEvent(QMouseEvent *event) override;
   void mouseMoveEvent(QMouseEvent *event) override;
   void mouseReleaseEvent(QMouseEvent *event) override;
   void mouseDoubleClickEvent(QMouseEvent* event) override;

   QAction* addMenuAction(QMenu *menu, QSignalMapper *map, const QString &text, int id);

   double scaledPosition(int p) { return (double) p * fQtScalingfactor; }

   QPoint scaledMousePoint(QMouseEvent *event);

   TCanvas *fCanvas = nullptr;

   QPainter *fPainter = nullptr;

   double fQtScalingfactor = 1.;

   QStatusBar  *fStatusBar = nullptr;

   bool              fMaskDoubleClick = false;
   double            fMousePosX = 0;    // mouse position in user coordinate when activate menu
   double            fMousePosY = 0;    // mouse position in user coordinate when activate menu

   TObject          *fMenuObj = nullptr;      // object use to fill menu
   TList            *fMenuMethods = nullptr;  // list of menu methods
   bool              fShowEventStatus = true;

};

#endif
