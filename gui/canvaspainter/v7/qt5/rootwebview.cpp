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

RootWebView::RootWebView(QWidget *parent) : QWebEngineView(parent)
{
   setPage(new RootWebPage());

   // connect(this, SIGNAL(javaScriptConsoleMessage(JavaScriptConsoleMessageLevel, const QString &, int, const QString
   // &)),
   //        this, SLOT(doConsole(JavaScriptConsoleMessageLevel, const QString &, int, const QString &)));

   // connect(this, &QWebEngineView::javaScriptConsoleMessage, this, &RootWebView::doConsole);
}

RootWebView::~RootWebView()
{
}
