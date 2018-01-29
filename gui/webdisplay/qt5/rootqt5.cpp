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
#include <qwebengineview.h>
#include <qtwebengineglobal.h>
#include <QThread>

// #include <QWebEngineUrlRequestJob>
#include <QWebEngineUrlRequestInterceptor>
// #include <QWebEngineUrlRequestInfo>
#include <QBuffer>
#include <QFile>

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

class ROOTRequestInterceptor : public QWebEngineUrlRequestInterceptor {
public:
   ROOTRequestInterceptor(QObject *p = Q_NULLPTR) : QWebEngineUrlRequestInterceptor(p) {}
   virtual void interceptRequest(QWebEngineUrlRequestInfo &info)
   {
      QUrl url = info.requestUrl();

      QByteArray ba = url.toString().toLatin1();

      printf("[%ld] Request intercepted %s\n", TThread::SelfId(), ba.data());
   }
};

extern "C" void webgui_start_browser_in_qt5(const char *url, void *http_serv, bool is_batch, unsigned width, unsigned height)
{
   // webgui_initapp();

   TString fullurl = UrlSchemeHandler::installHandler(TString(url), (THttpServer *)http_serv, !is_batch);

   if (is_batch) {
      RootWebPage *page = new RootWebPage();
      page->load(QUrl(fullurl.Data()));
   } else {
      RootWebView *view = new RootWebView(0, width, height);
      view->load(QUrl(fullurl.Data()));
      view->show();
   }
}

int main(int argc, char *argv[])
{
   QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);

   int argc2 = 1;
   char *argv2[10];
   argv2[0] = argv[0];

   // printf("[%ld] Start minimal app\n", TThread::SelfId());

   QApplication app(argc2, argv2);

   QtWebEngine::initialize();

   // QQmlApplicationEngine engine;
   // engine.load(QUrl(QStringLiteral("qrc:/main.qml")));
   // engine.load(QUrl("http://jsroot.gsi.de/dev/examples.htm"));

   TRint *d = new TRint("Rint", &argc, argv);
   TQt5Timer *timer = new TQt5Timer(10, kTRUE);
   timer->TurnOn();

   // use only for debugging or may be for redirection
   // QWebEngineProfile::defaultProfile()->setRequestInterceptor(new ROOTRequestInterceptor());

   // const QWebEngineUrlSchemeHandler* installed =
   // QWebEngineProfile::defaultProfile()->urlSchemeHandler(EXAMPLE_SCHEMA_HANDLER);

   // QWebEngineView *view = new QWebEngineView();
   // view->load(QUrl("http://jsroot.gsi.de/dev/examples.htm"));
   // view->show();

   d->Run();

   // while (true) {
   //   QApplication::processEvents();
   //   QApplication::sendPostedEvents();
   //   QThread::msleep(10);
   // }

   return 0;

   // return app.exec();
}
