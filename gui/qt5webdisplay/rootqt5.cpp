/// \file rootqt5.cpp
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

#include <QApplication>
// #include <QQmlApplicationEngine>
#include <QWebEngineView>
#include <qtwebengineglobal.h>
#include <QThread>
#include <QWebEngineSettings>

#include "TROOT.h"
#include "TApplication.h"
#include "TRint.h"
#include "TTimer.h"
#include "TThread.h"
#include "THttpServer.h"

#include <stdio.h>

#include "rootwebview.h"
#include "rootwebpage.h"
#include "rooturlschemehandler.h"

class TQt5Timer : public TTimer {
public:
   TQt5Timer(Long_t milliSec, Bool_t mode) : TTimer(milliSec, mode)
   {
      // construtor
   }
   virtual ~TQt5Timer()
   {
      // destructor
   }
   virtual void Timeout()
   {
      // timeout handler
      // used to process http requests in main ROOT thread

      QApplication::sendPostedEvents();
      QApplication::processEvents();
   }
};

QApplication *qapp = nullptr;
int qargc = 1;
char *qargv[10];

extern "C" void webgui_start_browser_in_qt5(const char *url, void *http_serv, bool is_batch, unsigned width, unsigned height)
{
   if (!qapp) {
      qargv[0] = gApplication->Argv(0);
      qapp = new QApplication(qargc, qargv);

      QtWebEngine::initialize();

      TQt5Timer *timer = new TQt5Timer(10, kTRUE);
      timer->TurnOn();
   }

   QString fullurl = UrlSchemeHandler::installHandler(url, (THttpServer *)http_serv);

   if (is_batch) {
      RootWebPage *page = new RootWebPage();
      page->settings()->resetAttribute(QWebEngineSettings::WebGLEnabled);
      page->settings()->resetAttribute(QWebEngineSettings::Accelerated2dCanvasEnabled);
      page->settings()->resetAttribute(QWebEngineSettings::PluginsEnabled);
      page->load(QUrl(fullurl));
   } else {
      RootWebView *view = new RootWebView(0, width, height);
      view->load(QUrl(fullurl));
      view->show();
   }
}

