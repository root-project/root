/// \file rootwebpage.h
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

#ifndef ROOT_UrlSchemeHandler
#define ROOT_UrlSchemeHandler

#include <QWebEngineUrlSchemeHandler>

class THttpServer;

class UrlRequestJobHolder : public QObject {
   Q_OBJECT

   QWebEngineUrlRequestJob *fRequest{nullptr};

public:
   UrlRequestJobHolder(QWebEngineUrlRequestJob *req);

   QWebEngineUrlRequestJob *req() { return fRequest; }

   void reset();

public slots:
   void onRequestDeleted(QObject *obj);
};

// ===============================================================


class UrlSchemeHandler : public QWebEngineUrlSchemeHandler {
   Q_OBJECT
protected:
   THttpServer *fServer; ///< server instance which should handle requests

   static int  gNumHandler;  ///< number of created handlers
   static THttpServer *gLastServer;  ///< keep pointer of last server

public:
   UrlSchemeHandler(QObject *p = Q_NULLPTR, THttpServer *server = Q_NULLPTR)
      : QWebEngineUrlSchemeHandler(p), fServer(server)
   {
   }

   virtual ~UrlSchemeHandler() {}

   virtual void requestStarted(QWebEngineUrlRequestJob *request);

   static QString installHandler(const QString &url, THttpServer *server);

};


#endif
