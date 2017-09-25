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

#include <QWebEngineUrlSchemeHandler>
#include <QWebEngineProfile>
#include <QWebEngineUrlRequestJob>
#include <QWebEngineUrlRequestInterceptor>
#include <QWebEngineUrlRequestInfo>
#include <QBuffer>
#include <QFile>

#include "TROOT.h"
#include "TApplication.h"
#include "TRint.h"
#include "TTimer.h"
#include "TThread.h"
#include "THttpServer.h"
#include "THttpCallArg.h"

#include <stdio.h>

#include "rootwebview.h"
#include "rootwebpage.h"

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

// THttpServer *server = 0;

int nhandler = 0;           // counter how many handler was created
void *last_http_server = 0; // remember which server was specified last time

// TODO: memory cleanup of these arguments
class TWebGuiCallArg : public THttpCallArg {
protected:
   QWebEngineUrlRequestJob *fRequest;
   int fDD;

public:
   TWebGuiCallArg(QWebEngineUrlRequestJob *req) : THttpCallArg(), fRequest(req), fDD(0) {}
   virtual ~TWebGuiCallArg()
   {
      if (fDD != 1)
         printf("FAAAAAAAAAAAAAIL %d\n", fDD);
   }

   void SendFile(const char *fname)
   {
      const char *mime = THttpServer::GetMimeType(fname);

      printf("Sending file %s\n", fname);

      QBuffer *buffer = new QBuffer;
      fRequest->connect(fRequest, SIGNAL(destroyed()), buffer, SLOT(deleteLater()));

      QFile file(fname);
      buffer->open(QIODevice::WriteOnly);
      if (file.open(QIODevice::ReadOnly)) {
         QByteArray arr = file.readAll();
         buffer->write(arr);
      }
      file.close();
      buffer->close();

      fRequest->reply(mime, buffer);
      fDD++;
   }

   virtual void HttpReplied()
   {
      if (Is404()) {
         printf("Request MISS %s %s\n", GetPathName(), GetFileName());

         fRequest->fail(QWebEngineUrlRequestJob::UrlNotFound);
         fDD++;
         // abort request
      } else if (IsFile()) {
         // send file
         SendFile((const char *)GetContent());
      } else {

         // printf("Reply %s %s typ:%s res:%ld\n", GetPathName(), GetFileName(), GetContentType(), GetContentLength());
         // if (GetContentLength()<100) printf("BODY:%s\n", (const char*) GetContent());

         QBuffer *buffer = new QBuffer;
         fRequest->connect(fRequest, SIGNAL(destroyed()), buffer, SLOT(deleteLater()));

         buffer->setData((const char *)GetContent(), GetContentLength());

         fRequest->reply(GetContentType(), buffer);
         fDD++;
      }
   }
};

class ROOTSchemeHandler : public QWebEngineUrlSchemeHandler {
protected:
   THttpServer *fServer; ///< server instance which should handle requests
public:
   ROOTSchemeHandler(QObject *p = Q_NULLPTR, THttpServer *server = Q_NULLPTR)
      : QWebEngineUrlSchemeHandler(p), fServer(server)
   {
   }

   virtual void requestStarted(QWebEngineUrlRequestJob *request)
   {

      QUrl url = request->requestUrl();

      QByteArray ba = url.toString().toLatin1();

      // printf("[%ld] Request started %s\n", TThread::SelfId(), ba.data());

      if (!fServer) {
         printf("HttpServer is not specified\n");
         return;
      }

      TWebGuiCallArg *arg = new TWebGuiCallArg(request);

      QString inp_path = url.path();
      QString inp_query = url.query();
      QString inp_method = request->requestMethod();

      TString fname;

      if (fServer->IsFileRequested(inp_path.toLatin1().data(), fname)) {

         arg->SendFile(fname.Data());
         delete arg;
         // process file
         return;
      }

      arg->SetPathAndFileName(inp_path.toLatin1().data());
      arg->SetQuery(inp_query.toLatin1().data());
      arg->SetMethod(inp_method.toLatin1().data());
      arg->SetTopName("webgui");

      fServer->SubmitHttp(arg);
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

extern "C" void webgui_start_browser_in_qt5(const char *url, void *http_serv, bool is_batch)
{
   // webgui_initapp();

   char protocol[100], fullurl[2000];
   bool create_handler = false;

   if (last_http_server != http_serv) {
      last_http_server = http_serv;
      create_handler = true;
      nhandler++;
   }

   snprintf(protocol, sizeof(protocol), "roothandler%d", nhandler);
   snprintf(fullurl, sizeof(fullurl), "%s%s", protocol, url);

   printf("Start %s\n", fullurl);

   if (create_handler) {
      const QByteArray protocol_name = QByteArray(protocol);
      ROOTSchemeHandler *handler = new ROOTSchemeHandler(Q_NULLPTR, (THttpServer *)http_serv);
      QWebEngineProfile::defaultProfile()->installUrlSchemeHandler(protocol_name, handler);
   }

   if (is_batch) {
      RootWebPage *page = new RootWebPage();
      page->load(QUrl(fullurl));
   } else {
      RootWebView *view = new RootWebView();
      view->load(QUrl(fullurl));
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
