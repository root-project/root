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

#include "rootwebview.h"
#include "rootwebpage.h"

#include <QMimeData>
#include <QDragEnterEvent>
#include <QDropEvent>

/** \class RootWebView
\ingroup qt6webdisplay
*/

RootWebView::RootWebView(QWidget *parent, unsigned width, unsigned height, int x, int y) :
   QWebEngineView(parent),
   fWidth(width),
   fHeight(height),
   fX(x),
   fY(y)
{
   setObjectName("RootWebView");

   setPage(new RootWebPage(this));

   connect(page(), &QWebEnginePage::windowCloseRequested, this, &RootWebView::onWindowCloseRequested);

   connect(page(), &QWebEnginePage::loadFinished /*   loadStarted */, this, &RootWebView::onLoadStarted);

   setAcceptDrops(true);

   if ((fX >= 0) || (fY >= 0)) move(fX > 0 ? fX : 0, fY > 0 ? fY : 0);

   // do not destroy view on close, one require some time to handle close events
   setAttribute( Qt::WA_DeleteOnClose, false );
}

QSize RootWebView::sizeHint() const
{
   if (fWidth && fHeight)
      return QSize(fWidth, fHeight);
   return QWebEngineView::sizeHint();
}

void RootWebView::dragEnterEvent( QDragEnterEvent *e )
{
   if (e->mimeData()->hasText())
      e->acceptProposedAction();
}


void RootWebView::dropEvent(QDropEvent* event)
{
   emit drop(event);
}

void RootWebView::closeEvent(QCloseEvent *)
{
   page()->runJavaScript("if (window && window.onqt6unload) window.onqt6unload();");
}

void RootWebView::onLoadStarted()
{
}

void RootWebView::onWindowCloseRequested()
{
   close();
}
