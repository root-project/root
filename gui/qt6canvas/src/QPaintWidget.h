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

class QPaintWidget : public QWidget {

   Q_OBJECT

public:
   QPaintWidget(QWidget *parent = nullptr);
   virtual ~QPaintWidget();

   /// returns canvas shown in the widget
   TCanvas *getCanvas() { return fCanvas; }

   void SetCanvas(TCanvas *canv) { fCanvas = canv; }

protected:
   void resizeEvent(QResizeEvent *event) override;

   TCanvas *fCanvas = nullptr;
};

#endif
