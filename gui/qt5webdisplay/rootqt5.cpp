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

#include <QApplication>
#include <QWebEngineView>
#include <qtwebengineglobal.h>
#include <QThread>
#include <QWebEngineSettings>
#include <QWebEngineProfile>

#include "TROOT.h"
#include "TApplication.h"
#include "TTimer.h"
#include "TEnv.h"
#include "TThread.h"
#include "THttpServer.h"

#include "rootwebview.h"
#include "rootwebpage.h"
#include "rooturlschemehandler.h"

#include <memory>

#include <ROOT/RWebDisplayHandle.hxx>
#include <ROOT/RMakeUnique.hxx>
#include <ROOT/TLogger.hxx>

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

      std::unique_ptr<RWebDisplayHandle> Display(const RWebDisplayArgs &args) override
      {
         if (args.IsHeadless())
            return nullptr;

         if (!qapp && !QApplication::instance()) {

            if (!gApplication) {
               R__ERROR_HERE("Qt5") << "NOT FOUND gApplication to create QApplication";
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
            Int_t interval = gEnv->GetValue("WebGui.Qt5Timer", 1);
            if (interval > 0) {
               fTimer = new TQt5Timer(interval, kTRUE);
               fTimer->TurnOn();
            }
         }

         std::unique_ptr<RootUrlSchemeHandler> handler;
         QString fullurl = QString(args.GetUrl().c_str());

         // if no server provided - normal HTTP will be allowed to use
         if (args.GetHttpServer()) {
            handler = std::make_unique<RootUrlSchemeHandler>(args.GetHttpServer(), fCounter++);
            fullurl = handler->MakeFullUrl(fullurl);
         }

         QWidget *qparent = (QWidget *) args.GetDriverData();

         if (!args.GetUrlOpt().empty()) {
            fullurl.append(QString("&"));
            fullurl.append(QString(args.GetUrlOpt().c_str()));
         }

         auto handle = std::make_unique<RQt5WebDisplayHandle>(fullurl.toLatin1().constData(), handler);

         if (args.IsHeadless()) {
            RootWebPage *page = new RootWebPage();
            page->settings()->resetAttribute(QWebEngineSettings::WebGLEnabled);
            page->settings()->resetAttribute(QWebEngineSettings::Accelerated2dCanvasEnabled);
            page->settings()->resetAttribute(QWebEngineSettings::PluginsEnabled);
            page->load(QUrl(fullurl));
         } else {
            RootWebView *view = new RootWebView(qparent, args.GetWidth(), args.GetHeight());
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
