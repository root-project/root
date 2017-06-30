/****************************************************************************
**
** Copyright (C) 2016 The Qt Company Ltd.
** Contact: http://www.qt.io/licensing/
**
** This file is part of the examples of the Qt Toolkit.
**
** $QT_BEGIN_LICENSE:BSD$
** You may use this file under the terms of the BSD license as follows:
**
** "Redistribution and use in source and binary forms, with or without
** modification, are permitted provided that the following conditions are
** met:
**   * Redistributions of source code must retain the above copyright
**     notice, this list of conditions and the following disclaimer.
**   * Redistributions in binary form must reproduce the above copyright
**     notice, this list of conditions and the following disclaimer in
**     the documentation and/or other materials provided with the
**     distribution.
**   * Neither the name of The Qt Company Ltd nor the names of its
**     contributors may be used to endorse or promote products derived
**     from this software without specific prior written permission.
**
**
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
** OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
** LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
** DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
** THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
** (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
** OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
**
** $QT_END_LICENSE$
**
****************************************************************************/

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

#include "rootwebview.h"

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

THttpServer *server = 0;


// TODO: memory cleanup of these arguments
class TWebGuiCallArg : public THttpCallArg {
protected:

   QWebEngineUrlRequestJob *fRequest;
   int  fDD;

public:
   TWebGuiCallArg(QWebEngineUrlRequestJob *req) : THttpCallArg(), fRequest(req), fDD(0) {  }
   virtual ~TWebGuiCallArg() { if (fDD!=1) printf("FAAAAAAAAAAAAAIL %d\n", fDD); }

   void SendFile(const char *fname) {
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

      fRequest->reply(mime, buffer); fDD++;
   }

   virtual void HttpReplied() {

      if (Is404()) {
         printf("Request MISS %s %s\n", GetPathName(), GetFileName());

         fRequest->fail(QWebEngineUrlRequestJob::UrlNotFound); fDD++;
         // abort request
      } else if (IsFile()) {
         // send file
         SendFile((const char *)GetContent());
      } else {

         // printf("Reply %s %s typ:%s res:%ld\n", GetPathName(), GetFileName(), GetContentType(), GetContentLength());
         // if (GetContentLength()<100) printf("BODY:%s\n", (const char*) GetContent());

         QBuffer *buffer = new QBuffer;
         fRequest->connect(fRequest, SIGNAL(destroyed()), buffer, SLOT(deleteLater()));

         buffer->setData((const char *) GetContent(), GetContentLength());

         fRequest->reply(GetContentType(), buffer); fDD++;
      }
   }

};

class ROOTSchemeHandler : public QWebEngineUrlSchemeHandler {
public:
   ROOTSchemeHandler(QObject *p = Q_NULLPTR) : QWebEngineUrlSchemeHandler(p) {}

   virtual void requestStarted(QWebEngineUrlRequestJob *request) {

      QUrl url = request->requestUrl();

      QByteArray ba = url.toString().toLatin1();

      // printf("[%ld] Request started %s\n", TThread::SelfId(), ba.data());

      if (server==0) {
         server = (THttpServer*) gROOT->ProcessLine("TWebGuiFactory::GetHttpServer()");
         printf("Get HTTP server %p\n", server);
         if (!server) {
            printf("FAIL to get server\n");
            return;
         }
      }

      TWebGuiCallArg* arg = new TWebGuiCallArg(request);

      QString inp_path = url.path();
      QString inp_query = url.query();
      QString inp_method = request->requestMethod();

      TString fname;

      if (server->IsFileRequested(inp_path.toLatin1().data(), fname)) {

         arg->SendFile(fname.Data());
         delete arg;
         // process file
         return;
      }

      arg->SetPathAndFileName(inp_path.toLatin1().data());
      arg->SetQuery(inp_query.toLatin1().data());
      arg->SetMethod(inp_method.toLatin1().data());
      arg->SetTopName("webgui");

      // TODO: POST buffer

      // printf("SUBMIT %s %s\n", arg->GetPathName(), arg->GetFileName());

      server->SubmitHttp(arg);
   }
};

class ROOTRequestInterceptor : public QWebEngineUrlRequestInterceptor {
public:
    ROOTRequestInterceptor(QObject *p = Q_NULLPTR) : QWebEngineUrlRequestInterceptor(p) {}
    virtual void interceptRequest(QWebEngineUrlRequestInfo &info) {

       QUrl url = info.requestUrl();

       QByteArray ba = url.toString().toLatin1();

       printf("[%ld] Request intercepted %s\n", TThread::SelfId(), ba.data());

    }
};

extern "C" void webgui_start_browser_new(const char *url)
{

   // webgui_initapp();

   printf("Start %s\n", url);

   RootWebView *view = new RootWebView();
   view->load(QUrl(url));
   view->show();
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

    //QQmlApplicationEngine engine;
    //engine.load(QUrl(QStringLiteral("qrc:/main.qml")));
    //engine.load(QUrl("http://jsroot.gsi.de/dev/examples.htm"));

    TRint* d = new TRint("Rint", &argc, argv);
    TQt5Timer *timer = new TQt5Timer(10, kTRUE);
    timer->TurnOn();


    // use only for debugging or may be for redirection
    //QWebEngineProfile::defaultProfile()->setRequestInterceptor(new ROOTRequestInterceptor());

    const QByteArray EXAMPLE_SCHEMA_HANDLER = QByteArray("example");

    ROOTSchemeHandler* handler = new ROOTSchemeHandler();

    QWebEngineProfile::defaultProfile()->installUrlSchemeHandler(EXAMPLE_SCHEMA_HANDLER, handler);

    // const QWebEngineUrlSchemeHandler* installed = QWebEngineProfile::defaultProfile()->urlSchemeHandler(EXAMPLE_SCHEMA_HANDLER);

    //QWebEngineView *view = new QWebEngineView();
    //view->load(QUrl("http://jsroot.gsi.de/dev/examples.htm"));
    //view->show();

    d->Run();

    //while (true) {
    //   QApplication::processEvents();
    //   QApplication::sendPostedEvents();
    //   QThread::msleep(10);
    // }

    return 0;

    // return app.exec();
}

