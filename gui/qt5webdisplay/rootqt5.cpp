/// \file rootqt5.cpp
/// \ingroup WebGui
/// \author Sergey Linev <S.Linev@gsi.de>
/// \date 2017-06-29
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
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
#include <QtGlobal>

#if QT_VERSION >= 0x050C00
#include <QWebEngineUrlScheme>
#endif

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
#include <ROOT/RLogger.hxx>

class TQt5Timer : public TTimer {
public:
   TQt5Timer(Long_t milliSec, Bool_t mode) : TTimer(milliSec, mode) {}

   /// timeout handler
   /// used to process all qt5 events in main ROOT thread
   void Timeout() override
   {
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
      std::unique_ptr<TQt5Timer> fTimer; ///< timer to process ROOT events
      std::unique_ptr<RootUrlSchemeHandler> fHandler; ///< specialized handler
   public:

      Qt5Creator() = default;

      virtual ~Qt5Creator()
      {
         /** Code executed during exit and sometime crashes.
          *  Disable it, while not clear if defaultProfile can be still used - seems to be not */
         // if (fHandler)
         //   QWebEngineProfile::defaultProfile()->removeUrlSchemeHandler(fHandler.get());
      }

      std::unique_ptr<RWebDisplayHandle> Display(const RWebDisplayArgs &args) override
      {
         if (args.IsHeadless())
            return nullptr;

         if (!qapp && !QApplication::instance()) {

            if (!gApplication) {
               R__ERROR_HERE("Qt5") << "NOT FOUND gApplication to create QApplication";
               return nullptr;
            }

            #if QT_VERSION >= 0x050C00
            QWebEngineUrlScheme scheme("rootscheme");
            scheme.setSyntax(QWebEngineUrlScheme::Syntax::HostAndPort);
            scheme.setDefaultPort(2345);
            scheme.setFlags(QWebEngineUrlScheme::SecureScheme);
            QWebEngineUrlScheme::registerScheme(scheme);
            #endif

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
               fTimer = std::make_unique<TQt5Timer>(interval, kTRUE);
               fTimer->TurnOn();
            }
         }

         QString fullurl = QString(args.GetFullUrl().c_str());

         // if no server provided - normal HTTP will be allowed to use
         if (args.GetHttpServer()) {
            if (!fHandler) {
               fHandler = std::make_unique<RootUrlSchemeHandler>();
               QWebEngineProfile::defaultProfile()->installUrlSchemeHandler("rootscheme", fHandler.get());
               QWebEngineProfile::defaultProfile()->connect(QWebEngineProfile::defaultProfile(), &QWebEngineProfile::downloadRequested,
                              [](QWebEngineDownloadItem *item) { item->accept(); });
            }

            fullurl = fHandler->MakeFullUrl(args.GetHttpServer(), fullurl);
         }

         QWidget *qparent = (QWidget *) args.GetDriverData();

         auto handle = std::make_unique<RQt5WebDisplayHandle>(fullurl.toLatin1().constData());

         if (args.IsHeadless()) {
            RootWebPage *page = new RootWebPage();
            #if QT_VERSION >= 0x050700
            page->settings()->resetAttribute(QWebEngineSettings::WebGLEnabled);
            page->settings()->resetAttribute(QWebEngineSettings::Accelerated2dCanvasEnabled);
            #endif
            page->settings()->resetAttribute(QWebEngineSettings::PluginsEnabled);
            page->load(QUrl(fullurl));
         } else {
            RootWebView *view = new RootWebView(qparent, args.GetWidth(), args.GetHeight(), args.GetX(), args.GetY());
            view->load(QUrl(fullurl));
            view->show();
         }

         return handle;
      }

   };

public:
   RQt5WebDisplayHandle(const std::string &url) : RWebDisplayHandle(url) {}

   static void AddCreator()
   {
      auto &entry = FindCreator("qt5");
      if (!entry)
         GetMap().emplace("qt5", std::make_unique<Qt5Creator>());
   }

};

struct RQt5CreatorReg {
   RQt5CreatorReg() { RQt5WebDisplayHandle::AddCreator(); }
} newRQt5CreatorReg;

}
}
