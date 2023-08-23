// Author: Sergey Linev <S.Linev@gsi.de>
// Date: 2017-06-29
// Warning: This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!

/*************************************************************************
 * Copyright (C) 1995-2023, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <QApplication>
#include <QWebEngineView>
#include <qtwebengineglobal.h>
#include <QWebEngineDownloadItem>

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
#include "TDirectory.h"

#include "rootwebview.h"
#include "rootwebpage.h"
#include "rooturlschemehandler.h"

#include <memory>

#include <ROOT/RWebDisplayHandle.hxx>
#include <ROOT/RWebWindowsManager.hxx>
#include <ROOT/RLogger.hxx>

QWebEngineUrlScheme gRootScheme("rootscheme");
QApplication *gOwnApplication = nullptr;
int gQt5HandleCounts = 0;
bool gProcEvents = false, gDoingShutdown = false;

void TestQt5Cleanup()
{
   if (gQt5HandleCounts == 0 && gOwnApplication && !gProcEvents && gDoingShutdown) {
      delete gOwnApplication;
      gOwnApplication = nullptr;
   }
}

class DummyObject : public TObject {
public:
   ~DummyObject() override
   {
      gDoingShutdown = true;
      TestQt5Cleanup();
   }

};

/** \class TQt5Timer
\ingroup qt5webdisplay
*/

class TQt5Timer : public TTimer {
public:
   TQt5Timer(Long_t milliSec, Bool_t mode) : TTimer(milliSec, mode) {}

   /// timeout handler
   /// used to process all qt5 events in main ROOT thread
   void Timeout() override
   {
      gProcEvents = true;
      QApplication::sendPostedEvents();
      QApplication::processEvents();
      gProcEvents = false;

   }
};

namespace ROOT {

/** \class RQt5WebDisplayHandle
\ingroup qt5webdisplay
*/

class RQt5WebDisplayHandle : public RWebDisplayHandle {
protected:

   RootWebView *fView{nullptr};  ///< pointer on widget, need to release when handle is destroyed

   class Qt5Creator : public Creator {
      int qargc{1};                 ///< arg counter
      char *qargv[2];               ///< arg values
      std::unique_ptr<TQt5Timer> fTimer; ///< timer to process ROOT events
      std::unique_ptr<RootUrlSchemeHandler> fHandler; ///< specialized handler
   public:

      Qt5Creator() = default;

      ~Qt5Creator() override
      {
         /** Code executed during exit and sometime crashes.
          *  Disable it, while not clear if defaultProfile can be still used - seems to be not */
         // if (fHandler)
         //   QWebEngineProfile::defaultProfile()->removeUrlSchemeHandler(fHandler.get());

         // do not try to destroy objects during exit
         fHandler.release();
         fTimer.release();
      }

      std::unique_ptr<RWebDisplayHandle> Display(const RWebDisplayArgs &args) override
      {
         if (!gOwnApplication && !QApplication::instance()) {

            if (!gApplication) {
               R__LOG_ERROR(QtWebDisplayLog()) << "Not found gApplication to create QApplication";
               return nullptr;
            }

            // initialize web engine only before creating QApplication
            QtWebEngine::initialize();

            qargv[0] = gApplication->Argv(0);
            qargv[1] = nullptr;

            gOwnApplication = new QApplication(qargc, qargv);

            // this is workaround to detect ROOT shutdown
            TDirectory::TContext ctxt; // preserve gDirectory
            auto dir = new TDirectory("dummy_qt5web_dir", "cleanup instance for qt5web");
            dir->GetList()->Add(new DummyObject());
            gROOT->GetListOfClosedObjects()->Add(dir);
         }

         // create timer to process Qt events from inside ROOT process events
         // very much improve performance, even when Qt event loop runs by QApplication normally
         if (!fTimer && !args.IsHeadless()) {
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

         QWidget *qparent = static_cast<QWidget *>(args.GetDriverData());

         auto handle = std::make_unique<RQt5WebDisplayHandle>(fullurl.toLatin1().constData());

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

            #if QT_VERSION >= 0x050900
            if (!pdffile.empty())
               QObject::connect(view->page(), &RootWebPage::pdfPrintingFinished, [&expired, &is_error](const QString &, bool is_ok) {
                  expired = 0; is_error = !is_ok;
               });
            #endif

            const std::string &page_content = args.GetPageContent();
            if (page_content.empty())
               view->load(QUrl(fullurl));
            else
               view->setHtml(QString::fromUtf8(page_content.data(), page_content.size()), QUrl("file:///batch_page.html"));

            // loop here until content is configured
            while ((--expired > 0) && !get_content && !is_error) {

               if (gSystem->ProcessEvents()) break; // interrupted, has to return

               gProcEvents = true;
               QApplication::sendPostedEvents();
               QApplication::processEvents();
               gProcEvents = false;

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
               gProcEvents = true;
               QApplication::sendPostedEvents();
               QApplication::processEvents();
               gProcEvents = false;
            }

         }

         return handle;
      }

   };

public:
   RQt5WebDisplayHandle(const std::string &url) : RWebDisplayHandle(url) { gQt5HandleCounts++; }

   ~RQt5WebDisplayHandle() override
   {
      // now view can be safely destroyed
      if (fView) {
         delete fView;
         fView = nullptr;
      }

      gQt5HandleCounts--;

      TestQt5Cleanup();
   }

   bool Resize(int width, int height) override
   {
      if (!fView)
         return false;
      fView->resize(QSize(width, height));
      return true;
   }

   static void AddCreator()
   {
      auto &entry = FindCreator("qt5");
      if (!entry)
         GetMap().emplace("qt5", std::make_unique<Qt5Creator>());
   }

};

struct RQt5CreatorReg {
   RQt5CreatorReg() {
      RQt5WebDisplayHandle::AddCreator();

      gRootScheme.setSyntax(QWebEngineUrlScheme::Syntax::HostAndPort);
      gRootScheme.setDefaultPort(2345);
      gRootScheme.setFlags(QWebEngineUrlScheme::SecureScheme);
      QWebEngineUrlScheme::registerScheme(gRootScheme);

   }
} newRQt5CreatorReg;

} // namespace ROOT
