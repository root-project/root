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
// #include <QQmlApplicationEngine>
#include <QWebEngineView>
#include <qtwebengineglobal.h>
#include <QThread>
#include <QWebEngineSettings>

#include "TROOT.h"
#include "TApplication.h"
#include "TRint.h"
#include "TTimer.h"
#include "TThread.h"
#include "THttpServer.h"

#include <stdio.h>

#include "rootwebview.h"
#include "rootwebpage.h"
#include "rooturlschemehandler.h"

#include <memory>

#include <ROOT/RWebDisplayHandle.hxx>
#include <ROOT/RMakeUnique.hxx>


class TQt5Timer : public TTimer {
public:
   TQt5Timer(Long_t milliSec, Bool_t mode) : TTimer(milliSec, mode)
   {
      // construtor
   }
   virtual ~TQt5Timer()
   {
      // destructor
   }
   virtual void Timeout()
   {
      // timeout handler
      // used to process http requests in main ROOT thread

      QApplication::sendPostedEvents();
      QApplication::processEvents();
   }
};

QApplication *qapp = nullptr;
int qargc = 1;
char *qargv[10];


namespace ROOT {
namespace Experimental {

class RQt5WebDisplayHandle : public RWebDisplayHandle {
protected:
   class Qt5Creator : public Creator {
   public:

      Qt5Creator() = default;

      virtual std::unique_ptr<RWebDisplayHandle>
      Make(THttpServer *serv, const std::string &url, bool batch, int width, int height)
      {
         if (batch)
            return nullptr;

         if (!qapp && !QApplication::instance()) {
            qargv[0] = gApplication->Argv(0);
            qapp = new QApplication(qargc, qargv);

            QtWebEngine::initialize();

            TQt5Timer *timer = new TQt5Timer(10, kTRUE);
            timer->TurnOn();
         }

         QString fullurl = UrlSchemeHandler::installHandler(QString(url.c_str()), serv);

         if (batch) {
            RootWebPage *page = new RootWebPage();
            page->settings()->resetAttribute(QWebEngineSettings::WebGLEnabled);
            page->settings()->resetAttribute(QWebEngineSettings::Accelerated2dCanvasEnabled);
            page->settings()->resetAttribute(QWebEngineSettings::PluginsEnabled);
            page->load(QUrl(fullurl));
         } else {
            RootWebView *view = new RootWebView(0, width, height);
            view->load(QUrl(fullurl));
            view->show();
         }

         return std::make_unique<RQt5WebDisplayHandle>(fullurl.toLatin1().constData());
      }
      virtual ~Qt5Creator() = default;
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
