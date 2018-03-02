/// \file rootwebview.cpp
/// \ingroup CanvasPainter ROOT7
/// \author Sergey Linev <S.Linev@gsi.de>
/// \date 2017-06-29
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
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

RootWebView::RootWebView(QWidget *parent, unsigned width, unsigned height) :
   QWebEngineView(parent),
   fWidth(width),
   fHeight(height)
{
   setPage(new RootWebPage());

   // connect(this, SIGNAL(javaScriptConsoleMessage(JavaScriptConsoleMessageLevel, const QString &, int, const QString
   // &)),
   //        this, SLOT(doConsole(JavaScriptConsoleMessageLevel, const QString &, int, const QString &)));

   // connect(this, &QWebEngineView::javaScriptConsoleMessage, this, &RootWebView::doConsole);

   connect(page(), &QWebEnginePage::windowCloseRequested, this, &RootWebView::onWindowCloseRequested);

   setAcceptDrops(true);
}

RootWebView::~RootWebView()
{
}

QSize RootWebView::sizeHint() const
{
   if (fWidth && fHeight) return QSize(fWidth, fHeight);
   return QWebEngineView::sizeHint();
}

void RootWebView::dragEnterEvent( QDragEnterEvent *e )
{
   if (e->mimeData()->hasText())
      e->acceptProposedAction();
}


void RootWebView::dropEvent(QDropEvent* event)
{
   printf("RootWebView drop event\n");
   emit drop(event);
}



void RootWebView::closeEvent(QCloseEvent *)
{
   page()->runJavaScript("if (window && window.onqt5unload) window.onqt5unload();");

   // printf("run javascript done\n");
}

void RootWebView::onWindowCloseRequested()
{
   close();
}
