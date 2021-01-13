// Author: Sergey Linev, GSI  13/01/2021

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef RCanvasWidget_H
#define RCanvasWidget_H

#include <QWidget>
#include <QWebEngineView>

namespace ROOT {
namespace Experimental {
class RCanvas;
}
}

class RCanvasWidget : public QWidget {

   Q_OBJECT

public:
   RCanvasWidget(QWidget *parent = nullptr);
   virtual ~RCanvasWidget();

   /// returns canvas shown in the widget
   ROOT::Experimental::RCanvas *getCanvas() { return fCanvas; }

protected:

   void resizeEvent(QResizeEvent *event) override;

   QWebEngineView *fView{nullptr};  ///< qt webwidget to show

   ROOT::Experimental::RCanvas *fCanvas{nullptr};
};

#endif
