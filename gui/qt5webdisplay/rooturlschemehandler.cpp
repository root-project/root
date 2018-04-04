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
#include <QFile>
#include <QWebEngineUrlRequestJob>
#include <QWebEngineProfile>

#include "THttpServer.h"
#include "THttpCallArg.h"

#include <stdio.h>

int UrlSchemeHandler::gNumHandler = 0;
THttpServer *UrlSchemeHandler::gLastServer = nullptr;

class TWebGuiCallArg : public THttpCallArg {
protected:
   QWebEngineUrlRequestJob *fRequest;

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
   }

   virtual void HttpReplied()
   {
      if (!fRequest) {
         printf("Qt5 Request already processed %s %s\n", GetPathName(), GetFileName());
         return;
      }

      if (Is404()) {
         printf("Request MISS %s %s\n", GetPathName(), GetFileName());

         fRequest->fail(QWebEngineUrlRequestJob::UrlNotFound);
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
      }

      fRequest = nullptr;
   }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

void UrlSchemeHandler::requestStarted(QWebEngineUrlRequestJob *request)
{
   QUrl url = request->requestUrl();

   QByteArray ba = url.toString().toLatin1();

   printf("Request started %s\n", ba.data());

   if (!fServer) {
      printf("HttpServer is not specified\n");
      return;
   }

   auto arg = std::make_shared<TWebGuiCallArg>(request);

   QString inp_path = url.path();
   QString inp_query = url.query();
   QString inp_method = request->requestMethod();

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

   fServer->SubmitHttp(arg);
}

/////////////////////////////////////////////////////////////////

TString UrlSchemeHandler::installHandler(const TString &url, THttpServer *server, bool use_openui)
{
   TString protocol, fullurl;
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

   return fullurl;
}
