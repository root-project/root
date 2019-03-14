/// \file rootqt5.cpp
/// \ingroup WebUI
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
         R__ERROR_HERE("Qt5") << "Qt5 request already processed path " << GetPathName() << " file " << GetFileName();
         return;
      }

      if (Is404()) {
         R__ERROR_HERE("Qt5") << "Qt5 request FAIL path " << GetPathName() << " file " << GetFileName();

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

RootUrlSchemeHandler::RootUrlSchemeHandler(THttpServer *server, int counter)
   : QWebEngineUrlSchemeHandler(), fServer(server)
{
   fProtocol = Form("roothandler%d", counter);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
/// Start processing of emulated HTTP request in WebEngine scheme handler
/// Either one reads file or redirect request to THttpServer

void RootUrlSchemeHandler::requestStarted(QWebEngineUrlRequestJob *request)
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

   // process file
   if (fServer->IsFileRequested(inp_path.toLatin1().data(), fname)) {
      arg->SendFile(fname.Data());
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
/// Returns fully qualified URL, required to open in QWindow

QString RootUrlSchemeHandler::MakeFullUrl(const QString &url)
{
   QString res = fProtocol;
   res.append(":");
   res.append(url);
   if (url.indexOf("?")<0)
      res.append("?");
   else
      res.append("&");
   res.append("platform=qt5&ws=rawlongpoll");
   return res;
}
