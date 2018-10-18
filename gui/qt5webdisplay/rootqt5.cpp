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
#include <QWebEngineView>
#include <qtwebengineglobal.h>
#include <QThread>
#include <QWebEngineSettings>
#include <QWebEngineProfile>

#include "TROOT.h"
#include "TApplication.h"
#include "TRint.h"
#include "TTimer.h"
#include "TThread.h"
#include "THttpServer.h"

#include "rootwebview.h"
#include "rootwebpage.h"
#include "rooturlschemehandler.h"

#include <memory>

#include <ROOT/RWebDisplayHandle.hxx>
#include <ROOT/RMakeUnique.hxx>

#include <stdio.h>

class TQt5Timer : public TTimer {
public:
   TQt5Timer(Long_t milliSec, Bool_t mode) : TTimer(milliSec, mode) {}
   virtual void Timeout()
   {
      // timeout handler
      // used to process all qt5 events in main ROOT thread
      QApplication::sendPostedEvents();
      QApplication::processEvents();
   }
};

namespace ROOT {
namespace Experimental {

class RQt5WebDisplayHandle : public RWebDisplayHandle {
protected:
   class Qt5Creator : public Creator {
      int fCounter{0}; ///< counter used to number handlers
      QApplication *qapp{nullptr};  ///< created QApplication
      int qargc{1};                 ///< arg counter
      char *qargv[10];              ///< arg values
      bool fInitEngine{false};      ///< does engine was initialized
      TQt5Timer *fTimer{nullptr};   ///< timer to process ROOT events
   public:

      Qt5Creator() = default;

      std::unique_ptr<RWebDisplayHandle>
      ShowURL(const std::string &where, THttpServer *serv, const std::string &url, bool batch, int width, int height) override
      {
         if (batch)
            return nullptr;

         if (!qapp && !QApplication::instance()) {

            if (!gApplication) {
               printf("NOT FOUND gApplication to create QApplication\n");
               return nullptr;
            }

            qargv[0] = gApplication->Argv(0);
            qargv[1] = nullptr;
            qapp = new QApplication(qargc, qargv);
         }

         if (!fInitEngine) {
            QtWebEngine::initialize();
            fInitEngine = true;
         }

         if (!fTimer) {
            fTimer = new TQt5Timer(10, kTRUE);
            fTimer->TurnOn();
         }

         std::unique_ptr<RootUrlSchemeHandler> handler;
         QString fullurl = QString(url.c_str());

         // if no server provided - normal HTTP will be allowed to use
         if (serv) {
            handler = std::make_unique<RootUrlSchemeHandler>(serv, fCounter++);
            fullurl = handler->MakeFullUrl(fullurl);
         }

         QWidget *qparent = nullptr;

         auto pos = where.find("qprnt:");
         if (pos != std::string::npos) {
            long long unsigned value = 0;
            sscanf(where.c_str() + pos + 6, "%llu", &value);
            qparent = (QWidget *) value;
         }

         pos = where.find("url:");
         if (pos != std::string::npos)
            fullurl.append(QString(where.substr(pos+4).c_str()));

         auto handle = std::make_unique<RQt5WebDisplayHandle>(fullurl.toLatin1().constData(), handler);

         if (batch) {
            RootWebPage *page = new RootWebPage();
            page->settings()->resetAttribute(QWebEngineSettings::WebGLEnabled);
            page->settings()->resetAttribute(QWebEngineSettings::Accelerated2dCanvasEnabled);
            page->settings()->resetAttribute(QWebEngineSettings::PluginsEnabled);
            page->load(QUrl(fullurl));
         } else {
            RootWebView *view = new RootWebView(qparent, width, height);
            view->load(QUrl(fullurl));
            view->show();
         }

         return handle;
      }

      virtual ~Qt5Creator()
      {
         if (fTimer) {
            fTimer->TurnOff();
            delete fTimer;
         }

      }
   };

   std::unique_ptr<RootUrlSchemeHandler> fHandler;

public:
   RQt5WebDisplayHandle(const std::string &url, std::unique_ptr<RootUrlSchemeHandler> &handler)
      : RWebDisplayHandle(url)
   {
      std::swap(fHandler, handler);
      if (fHandler)
         QWebEngineProfile::defaultProfile()->installUrlSchemeHandler(QByteArray(fHandler->GetProtocol()), fHandler.get());
   }

   static void AddCreator()
   {
      auto &entry = FindCreator("qt5");
      if (!entry)
         GetMap().emplace("qt5", std::make_unique<Qt5Creator>());
   }

   virtual ~RQt5WebDisplayHandle()
   {
      if (fHandler)
         QWebEngineProfile::defaultProfile()->removeUrlSchemeHandler(fHandler.get());
   }
};

struct RQt5CreatorReg {
   RQt5CreatorReg() { RQt5WebDisplayHandle::AddCreator(); }
} newRQt5CreatorReg;

}
}
