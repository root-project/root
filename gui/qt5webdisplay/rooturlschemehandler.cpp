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


#include "rooturlschemehandler.h"

#include <QBuffer>
#include <QByteArray>
#include <QFile>
#include <QWebEngineUrlRequestJob>
#include <QWebEngineProfile>

#include "THttpServer.h"
#include "THttpCallArg.h"
#include "TThread.h"

#include <stdio.h>

int UrlSchemeHandler::gNumHandler = 0;
THttpServer *UrlSchemeHandler::gLastServer = nullptr;

TRequestHolder::TRequestHolder(QWebEngineUrlRequestJob *req) : QObject(), fRequest(req)
{
   if (fRequest)
      connect(fRequest, &QObject::destroyed, this, &TRequestHolder::onRequestDeleted);
}

void TRequestHolder::onRequestDeleted(QObject *obj)
{
   if (fRequest == obj)
      fRequest = nullptr;
}

void TRequestHolder::reset()
{
   if (fRequest)
      disconnect(fRequest, &QObject::destroyed, this, &TRequestHolder::onRequestDeleted);
   fRequest = nullptr;
}


// ===================================================================


class TWebGuiCallArg : public THttpCallArg {

protected:
   TRequestHolder fRequest;

public:
   explicit TWebGuiCallArg(QWebEngineUrlRequestJob *req = nullptr) : THttpCallArg(), fRequest(req) {}

   virtual ~TWebGuiCallArg()
   {
      printf("Destroy TWebGuiCallArg %p %s %s %s\n", this, GetPathName(), GetFileName(), GetQuery());
   }

   void SendFile(const char *fname)
   {
      const char *mime = THttpServer::GetMimeType(fname);

      printf("Sending file %s\n", fname);

      QBuffer *buffer = new QBuffer;

      QFile file(fname);
      buffer->open(QIODevice::WriteOnly);
      if (file.open(QIODevice::ReadOnly)) {
         QByteArray arr = file.readAll();
         buffer->write(arr);
      }
      file.close();
      buffer->close();

      buffer->connect(buffer, &QIODevice::aboutToClose, buffer, &QObject::deleteLater);

      QWebEngineUrlRequestJob *req = fRequest.req();

      if (req) {
         req->reply(mime, buffer);
         fRequest.reset();
      }
   }

   virtual void HttpReplied()
   {
      QWebEngineUrlRequestJob *req = fRequest.req();

      if (!req) {
         printf("Qt5 Request already processed %s %s\n", GetPathName(), GetFileName());
         return;
      }

      if (Is404()) {
         printf("Request MISS %s %s\n", GetPathName(), GetFileName());

         req->fail(QWebEngineUrlRequestJob::UrlNotFound);
         // abort request
      } else if (IsFile()) {
         // send file
         SendFile((const char *)GetContent());
      } else {

         printf("Qt5 Reply %s %s typ:%s res:%ld\n", GetPathName(), GetFileName(), GetContentType(), GetContentLength());
         // if (GetContentLength()<100) printf("BODY:%s\n", (const char*) GetContent());

         QBuffer *buffer = new QBuffer;
         // fRequest->connect(fRequest, SIGNAL(destroyed()), buffer, SLOT(deleteLater()));

         // QByteArray arr((const char *)GetContent(), GetContentLength());

         buffer->open(QIODevice::WriteOnly);

         buffer->write((const char *)GetContent(), GetContentLength());

         buffer->close();

         buffer->connect(buffer, &QIODevice::aboutToClose, buffer, &QObject::deleteLater);

         req->reply(GetContentType(), buffer);

         printf("THRD: %ld JOB: %p Qt5 Reply send res:%ld\n", (long)TThread::SelfId(), req, GetContentLength());
      }

      fRequest.reset();
   }
};



///////////////////////////////////////////////////////////////////////////////////////////////////

void UrlSchemeHandler::requestStarted(QWebEngineUrlRequestJob *request)
{
   QUrl url = request->requestUrl();

   printf("Request started %p %s\n", request, url.toString().toLatin1().data());

   if (!fServer) {
      printf("HttpServer is not specified\n");
      return;
   }

   QString inp_path = url.path();
   QString inp_query = url.query();
   QString inp_method = request->requestMethod();

   std::string qq = inp_query.toLatin1().data();
   if ((qq.find("c22526f6f744d6f64656c3322") != std::string::npos) && false) {
      printf("FIND SPECIAL MSG\n");

      // fRequest->connect(fRequest, SIGNAL(destroyed()), buffer, SLOT(deleteLater()));

      // buffer->connect(buffer, &QIODevice::aboutToClose, buffer, &QObject::deleteLater);


      const char *fname = "testPanel.C";

      printf("Sending file %s\n", fname);

      QBuffer *buffer = new QBuffer;

      QFile file(fname);
      buffer->open(QIODevice::WriteOnly);
      if (file.open(QIODevice::ReadOnly)) {
         QByteArray arr = file.readAll();
         buffer->write(arr);
      }
      file.close();
      buffer->close();

      buffer->connect(buffer, &QIODevice::aboutToClose, buffer, &QObject::deleteLater);

      request->reply("text/plain", buffer);
      return;
   }


   auto arg = std::make_shared<TWebGuiCallArg>(request);

   TString fname;

   if (fServer->IsFileRequested(inp_path.toLatin1().data(), fname)) {
      arg->SendFile(fname.Data());
      // process file
      return;
   }

   arg->SetPathAndFileName(inp_path.toLatin1().data());
   arg->SetQuery(inp_query.toLatin1().data());
   arg->SetMethod(inp_method.toLatin1().data());
   arg->SetTopName("webgui");

   // connect(request, &QObject::destroyed, this, &UrlSchemeHandler::onJobDeleted);

   // can process immediately - function called in main thread
   fServer->SubmitHttp(arg, kTRUE);

   printf("Finish requestStarted %p\n", request);
}

/////////////////////////////////////////////////////////////////

QString UrlSchemeHandler::installHandler(const QString &url_, THttpServer *server, bool use_openui)
{
   TString protocol, fullurl, url(url_.toLatin1().data());
   bool create_handler = false;

   if (gLastServer != server) {
      gLastServer = server;
      create_handler = true;
      gNumHandler++;
   }

   const char *suffix = url.Index("?") != kNPOS ? "&" : "?";

   protocol.Form("roothandler%d", gNumHandler);
   fullurl.Form("%s://dummy:8080%s%sqt5%s", protocol.Data(), url.Data(), suffix, (use_openui ? "" : "&noopenui"));

   if (create_handler) {
      const QByteArray protocol_name = QByteArray(protocol.Data());
      UrlSchemeHandler *handler = new UrlSchemeHandler(Q_NULLPTR, server);
      QWebEngineProfile::defaultProfile()->installUrlSchemeHandler(protocol_name, handler);
   }

   return QString(fullurl.Data());
}
