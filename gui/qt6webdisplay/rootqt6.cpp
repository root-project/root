// Author: Sergey Linev <S.Linev@gsi.de>
// Date: 2017-06-29
// Warning: This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <QApplication>
#include <QWebEngineView>
#include <qtwebenginecoreglobal.h>
#include <QWebEngineDownloadRequest>
// #include <qtwebenginequickglobal.h>

#include <QThread>
#include <QWebEngineSettings>
#include <QWebEngineProfile>
#include <QtGlobal>

#include <QWebEngineUrlScheme>

#include "TROOT.h"
#include "TApplication.h"
#include "TTimer.h"
#include "TEnv.h"
#include "TThread.h"
#include "THttpServer.h"
#include "TSystem.h"

#include "../qt5webdisplay/rootwebview.h"
#include "../qt5webdisplay/rootwebpage.h"
#include "../qt5webdisplay/rooturlschemehandler.h"

#include <memory>

#include <ROOT/RWebDisplayHandle.hxx>
#include <ROOT/RLogger.hxx>

/** \class TQt6Timer
\ingroup qt6webdisplay
*/

class TQt6Timer : public TTimer {
public:
   TQt6Timer(Long_t milliSec, Bool_t mode) : TTimer(milliSec, mode) {}

   /// timeout handler
   /// used to process all qt6 events in main ROOT thread
   void Timeout() override
   {
      QApplication::sendPostedEvents();
      QApplication::processEvents();
   }
};

namespace ROOT {
namespace Experimental {

/** \class RQt6WebDisplayHandle
\ingroup qt6webdisplay
*/

class RQt6WebDisplayHandle : public RWebDisplayHandle {
protected:

   RootWebView *fView{nullptr};  ///< pointer on widget, need to release when handle is destroyed

   class Qt6Creator : public Creator {
      int fCounter{0}; ///< counter used to number handlers
      QApplication *qapp{nullptr};  ///< created QApplication
      int qargc{1};                 ///< arg counter
      char *qargv[2];               ///< arg values
      std::unique_ptr<TQt6Timer> fTimer; ///< timer to process ROOT events
      std::unique_ptr<RootUrlSchemeHandler> fHandler; ///< specialized handler
   public:

      Qt6Creator() = default;

      virtual ~Qt6Creator()
      {
         /** Code executed during exit and sometime crashes.
          *  Disable it, while not clear if defaultProfile can be still used - seems to be not */
         // if (fHandler)
         //   QWebEngineProfile::defaultProfile()->removeUrlSchemeHandler(fHandler.get());

         R__LOG_DEBUG(0, QtWebDisplayLog()) << "Deleting Qt6Creator";
      }

      std::unique_ptr<RWebDisplayHandle> Display(const RWebDisplayArgs &args) override
      {
         if (!qapp && !QApplication::instance()) {

            if (!gApplication) {
               R__LOG_ERROR(QtWebDisplayLog()) << "Not found gApplication to create QApplication";
               return nullptr;
            }

            // initialize web engine only before creating QApplication
            // QtWebEngineQuick::initialize();

            QWebEngineUrlScheme scheme("rootscheme");
            scheme.setSyntax(QWebEngineUrlScheme::Syntax::HostAndPort);
            scheme.setDefaultPort(2345);
            scheme.setFlags(QWebEngineUrlScheme::SecureScheme);
            QWebEngineUrlScheme::registerScheme(scheme);

            qargv[0] = gApplication->Argv(0);
            qargv[1] = nullptr;

            qapp = new QApplication(qargc, qargv);
         }

         // create timer to process Qt events from inside ROOT process events
         // very much improve performance, even when Qt even loop runs by QApplication normally
         if (!fTimer && !args.IsHeadless()) {
            Int_t interval = gEnv->GetValue("WebGui.Qt5Timer", 1);
            if (interval > 0) {
               fTimer = std::make_unique<TQt6Timer>(interval, kTRUE);
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
                              [](QWebEngineDownloadRequest *request) { request->accept(); });
            }

            fullurl = fHandler->MakeFullUrl(args.GetHttpServer(), fullurl);
         }

         QWidget *qparent = static_cast<QWidget *>(args.GetDriverData());

         auto handle = std::make_unique<RQt6WebDisplayHandle>(fullurl.toLatin1().constData());

         RootWebView *view = new RootWebView(qparent, args.GetWidth(), args.GetHeight(), args.GetX(), args.GetY());

         if (!args.IsHeadless()) {
            if (!qparent) handle->fView = view;
            view->load(QUrl(fullurl));
            view->show();
         } else {

            int tmout_sec = 30, expired = tmout_sec * 100;
            bool load_finished = false, did_try = false, get_content = false, is_error = false;
            std::string content, pdffile;

            if (!args.GetExtraArgs().empty() && (args.GetExtraArgs().find("--print-to-pdf=")==0))
               pdffile = args.GetExtraArgs().substr(15);

            QObject::connect(view, &RootWebView::loadFinished, [&load_finished, &is_error](bool is_ok) {
               load_finished = true; is_error = !is_ok;
            });

            if (!pdffile.empty())
               QObject::connect(view->page(), &RootWebPage::pdfPrintingFinished, [&expired, &is_error](const QString &, bool is_ok) {
                  expired = 0; is_error = !is_ok;
               });

            const std::string &page_content = args.GetPageContent();
            if (page_content.empty())
               view->load(QUrl(fullurl));
            else
               view->setHtml(QString::fromUtf8(page_content.data(), page_content.size()), QUrl("file:///batch_page.html"));

            // loop here until content is configured
            while ((--expired > 0) && !get_content && !is_error) {

               if (gSystem->ProcessEvents()) break; // interrupted, has to return

               QApplication::sendPostedEvents();
               QApplication::processEvents();

               if (load_finished && !did_try) {
                  did_try = true;

                  if (pdffile.empty()) {
                     view->page()->toHtml([&get_content, &content](const QString& res) {
                        get_content = true;
                        content = res.toLatin1().constData();
                     });
                  } else {
                     view->page()->printToPdf(QString::fromUtf8(pdffile.data(), pdffile.size()));
                     #if QT_VERSION < 0x050900
                     expired = 5; // no signal will be produced, just wait short time and break loop
                     #endif
                  }
               }

               gSystem->Sleep(10); // only 10 ms sleep
            }

            if(get_content)
               handle->SetContent(content);

            // delete view and process events
            delete view;

            for (expired=0;expired<100;++expired) {
               QApplication::sendPostedEvents();
               QApplication::processEvents();
            }

         }

         return handle;
      }

   };

public:
   RQt6WebDisplayHandle(const std::string &url) : RWebDisplayHandle(url) {}

   virtual ~RQt6WebDisplayHandle()
   {
      // now view can be safely destroyed
      if (fView) {
         delete fView;
         fView = nullptr;
      }
   }

   static void AddCreator()
   {
      auto &entry = FindCreator("qt6");
      if (!entry)
         GetMap().emplace("qt6", std::make_unique<Qt6Creator>());
   }

};

struct RQt6CreatorReg {
   RQt6CreatorReg() { RQt6WebDisplayHandle::AddCreator(); }
} newRQt6CreatorReg;

}
}
