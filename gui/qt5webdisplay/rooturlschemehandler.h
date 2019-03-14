/// \file rootwebpage.h
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


class RootUrlSchemeHandler : public QWebEngineUrlSchemeHandler {
   Q_OBJECT
protected:

   QString fProtocol;

   THttpServer *fServer{nullptr}; ///< server instance which should handle requests

public:
   RootUrlSchemeHandler(THttpServer *server = nullptr, int counter = 0);

   virtual ~RootUrlSchemeHandler() = default;

   QByteArray GetProtocol() const { return QByteArray(fProtocol.toLatin1().constData(), fProtocol.length()); }

   QString MakeFullUrl(const QString &url);

   virtual void requestStarted(QWebEngineUrlRequestJob *request);
};


#endif
