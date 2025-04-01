// Author: Sergey Linev <S.Linev@gsi.de>
// Date: 2017-06-29
// Warning: This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!

/*************************************************************************
 * Copyright (C) 1995-2023, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RootWebView
#define ROOT_RootWebView

#include <QWebEngineView>

class RootWebView : public QWebEngineView {
   Q_OBJECT
protected:
   unsigned fWidth{0}, fHeight{0};
   int fX{0}, fY{0};

   void closeEvent(QCloseEvent *) override;

   void dropEvent(QDropEvent *) override;
   void dragEnterEvent(QDragEnterEvent *) override;

public slots:
   void onLoadStarted();

   void onWindowCloseRequested();

signals:

   void drop(QDropEvent* event);

public:
   RootWebView(QWidget *parent = nullptr, unsigned width = 0, unsigned height = 0, int x = -1, int y = -1);
   virtual ~RootWebView() = default;

   QSize  sizeHint() const override;
};

#endif
