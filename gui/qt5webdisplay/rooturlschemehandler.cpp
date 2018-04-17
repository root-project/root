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

#include <ROOT/TLogger.hxx>

#include "THttpServer.h"
#include "THttpCallArg.h"
#include "TBase64.h"

int UrlSchemeHandler::gNumHandler = 0;
THttpServer *UrlSchemeHandler::gLastServer = nullptr;


/////////////////////////////////////////////////////////////////////////////////////
/// Class UrlRequestJobHolder
/// Required to monitor state of QWebEngineUrlRequestJob
/// Qt can delete object at any time, therefore one connects destroy signal
/// from the request to clear pointer
////////////////////////////////////////////////////////////////////////////////////

UrlRequestJobHolder::UrlRequestJobHolder(QWebEngineUrlRequestJob *req) : QObject(), fRequest(req)
{
   if (fRequest)
      connect(fRequest, &QObject::destroyed, this, &UrlRequestJobHolder::onRequestDeleted);
}

void UrlRequestJobHolder::onRequestDeleted(QObject *obj)
{
   if (fRequest == obj)
      fRequest = nullptr;
}

void UrlRequestJobHolder::reset()
{
   if (fRequest)
      disconnect(fRequest, &QObject::destroyed, this, &UrlRequestJobHolder::onRequestDeleted);
   fRequest = nullptr;
}

// ===================================================================

class TWebGuiCallArg : public THttpCallArg {

protected:
   UrlRequestJobHolder fRequest;

public:
   explicit TWebGuiCallArg(QWebEngineUrlRequestJob *req = nullptr) : THttpCallArg(), fRequest(req) {}

   virtual ~TWebGuiCallArg() {}

   void SendFile(const char *fname)
   {
      const char *mime = THttpServer::GetMimeType(fname);

      QBuffer *buffer = new QBuffer;

      QFile file(fname);
      buffer->open(QIODevice::WriteOnly);
      if (file.open(QIODevice::ReadOnly)) {
         QByteArray arr = file.readAll();
         buffer->write(arr);
      }
      file.close();
      buffer->close();

      QWebEngineUrlRequestJob *req = fRequest.req();

      if (req) {
         buffer->connect(req, &QObject::destroyed, buffer, &QObject::deleteLater);
         req->reply(mime, buffer);
         fRequest.reset();
      }
   }

   virtual void HttpReplied()
   {
      QWebEngineUrlRequestJob *req = fRequest.req();

      if (!req) {
         R__ERROR_HERE("webgui") << "Qt5 request already processed path " << GetPathName() << " file " << GetFileName();
         return;
      }

      if (Is404()) {
         R__ERROR_HERE("webgui") << "Qt5 request FAIL path " << GetPathName() << " file " << GetFileName();

         req->fail(QWebEngineUrlRequestJob::UrlNotFound);
         // abort request
      } else if (IsFile()) {
         // send file
         SendFile((const char *)GetContent());
      } else {

         QBuffer *buffer = new QBuffer;

         buffer->open(QIODevice::WriteOnly);

         buffer->write((const char *)GetContent(), GetContentLength());

         buffer->close();

         buffer->connect(req, &QObject::destroyed, buffer, &QObject::deleteLater);

         req->reply(GetContentType(), buffer);
      }

      fRequest.reset();
   }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

void UrlSchemeHandler::requestStarted(QWebEngineUrlRequestJob *request)
{
   QUrl url = request->requestUrl();

   if (!fServer) {
      R__ERROR_HERE("webgui") << "Server not specified when request is started";
      request->fail(QWebEngineUrlRequestJob::UrlNotFound);
      return;
   }

   QString inp_path = url.path();
   QString inp_query = url.query();
   QString inp_method = request->requestMethod();

   auto arg = std::make_shared<TWebGuiCallArg>(request);

   TString fname;

   if (fServer->IsFileRequested(inp_path.toLatin1().data(), fname)) {
      arg->SendFile(fname.Data());
      // process file
      return;
   }

   // Analyze and cut post data as soon as possible
   TString query = inp_query.toLatin1().data();
   Int_t pos = query.Index("&post=");
   if (pos != kNPOS) {
      TString buf = TBase64::Decode(query.Data() + pos + 6);
      arg->SetPostData(std::string(buf.Data()));
      query.Resize(pos);
   }

   arg->SetPathAndFileName(inp_path.toLatin1().data());
   arg->SetQuery(query.Data());
   arg->SetMethod(inp_method.toLatin1().data());
   arg->SetTopName("webgui");

   // can process immediately - function called in main thread
   fServer->SubmitHttp(arg, kTRUE);
}

/////////////////////////////////////////////////////////////////

QString UrlSchemeHandler::installHandler(const QString &url_, THttpServer *server)
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
   fullurl.Form("%s://rootserver.local%s%splatform=qt5&ws=rawlongpoll", protocol.Data(), url.Data(), suffix);

   if (create_handler) {
      const QByteArray protocol_name = QByteArray(protocol.Data());
      UrlSchemeHandler *handler = new UrlSchemeHandler(Q_NULLPTR, server);
      QWebEngineProfile::defaultProfile()->installUrlSchemeHandler(protocol_name, handler);
   }

   return QString(fullurl.Data());
}
