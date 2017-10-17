/// \file TWebDisplayManager.cxx
/// \ingroup WebGui ROOT7
/// \author Sergey Linev <s.linev@gsi.de>
/// \date 2017-10-16
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/TWebDisplayManager.hxx"

#include <cassert>
#include <cstdio>

#include "THttpServer.h"
#include "THttpEngine.h"

#include "TSystem.h"
#include "TRandom.h"
#include "TString.h"



std::shared_ptr<ROOT::Experimental::TWebDisplayManager> ROOT::Experimental::TWebDisplayManager::sInstance;

//const std::shared_ptr<ROOT::Experimental::TWebDisplayManager> &ROOT::Experimental::TWebDisplayManager::Instance()
//{
//   static std::shared_ptr<ROOT::Experimental::TWebDisplayManager> sManager;
//   return sManager;
//}

//static std::shared_ptr<ROOT::Experimental::TWebDisplayManager> ROOT::Experimental::TWebDisplayManager::Create()
//{
//   return std::make_shared<ROOT::Experimental::TWebDisplayManager>();
//}

ROOT::Experimental::TWebDisplayManager::~TWebDisplayManager()
{
   if (fServer) {
      delete fServer;
      fServer = 0;
   }
}



bool ROOT::Experimental::TWebDisplayManager::CreateHttpServer(bool with_http)
{
   if (!fServer)
      fServer = new THttpServer("dummy");

   if (!with_http || (fAddr.length() > 0))
      return true;

   // gServer = new THttpServer("http:8080?loopback&websocket_timeout=10000");

   int http_port = 0;
   const char *ports = gSystem->Getenv("WEBGUI_PORT");
   if (ports)
      http_port = std::atoi(ports);
   if (!http_port)
      gRandom->SetSeed(0);

   for (int ntry = 0; ntry < 100; ++ntry) {
      if (!http_port)
         http_port = (int)(8800 + 1000 * gRandom->Rndm(1));

      // TODO: ensure that port can be used
      // TODO: replace TString::Format with more adequate implementation like https://stackoverflow.com/questions/4668760
      if (fServer->CreateEngine(TString::Format("http:%d?websocket_timeout=10000", http_port))) {
         fAddr = "http://localhost:";
         fAddr.append(std::to_string(http_port));
         return true;
      }

      http_port = 0;
   }

   return false;
}

std::shared_ptr<ROOT::Experimental::TWebDisplay> ROOT::Experimental::TWebDisplayManager::CreateDisplay()
{
   std::shared_ptr<ROOT::Experimental::TWebDisplay> display = std::make_shared<ROOT::Experimental::TWebDisplay>();

   display->SetId(++fIdCnt); // set unique ID

   fDisplays.push_back(display);

   display->fMgr = sInstance;

   return std::move(display);
}

void ROOT::Experimental::TWebDisplayManager::CloseDisplay(ROOT::Experimental::TWebDisplay *display)
{
   // TODO: close all active connections of the display

   if (display->fWSHandler)
      fServer->Unregister((THttpWSHandler *) display->fWSHandler);

   for (auto displ = fDisplays.begin(); displ != fDisplays.end(); displ++) {
      if (displ->get() == display) {
         fDisplays.erase(displ);
         break;
      }
   }
}


//////////////////////////////////////////////////////////////////////////
/// Create new display for the canvas
/// Parameter \par where specified  which program could be used for display creation
/// Possible values:
///
///      cef - Chromium Embeded Framework, local display, local communication
///      qt5 - Qt5 WebEngine (when running via rootqt5), local display, local communication
///  browser - default system web-browser, communication via random http port from range 8800 - 9800
///  <prog> - any program name which will be started instead of default browser, like firefox or /usr/bin/opera
///           one could also specify $url in program name, which will be replaced with canvas URL
///  native - either any available local display or default browser
///
///  Canvas can be displayed in several different places

bool ROOT::Experimental::TWebDisplayManager::Show(ROOT::Experimental::TWebDisplay *display, const std::string &where, bool first_time)
{

   if (!CreateHttpServer()) {
      assert("Fail to create server");
      return false;
   }

   THttpWSHandler *handler = (THttpWSHandler *) display->fWSHandler;

   if (first_time)
      fServer->Register("/web7gui", handler);

   TString addr;

   bool is_native = where.empty() || (where == "native"), is_qt5 = (where == "qt5"), ic_cef = (where == "cef");

   Func_t symbol_qt5 = gSystem->DynFindSymbol("*", "webgui_start_browser_in_qt5");

   // TODO: batch mode and other URL parameters belong to the TCanvas
   bool batch_mode = false;

   if (symbol_qt5 && (is_native || is_qt5)) {
      typedef void (*FunctionQt5)(const char *, void *, bool);

      addr.Form("://dummy:8080/web7gui/%s/draw.htm?longpollcanvas&no_root_json%s&qt5", handler->GetName(),
                (batch_mode ? "&batch_mode" : ""));
      // addr.Form("example://localhost:8080/Canvases/%s/draw.htm", Canvas()->GetName());

      printf("Show canvas in Qt5 window:  %s\n", addr.Data());

      FunctionQt5 func = (FunctionQt5)symbol_qt5;
      func(addr.Data(), fServer, batch_mode);
      return false;
   }

   // TODO: one should try to load CEF libraries only when really needed
   // probably, one should create separate DLL with CEF-related code
   Func_t symbol_cef = gSystem->DynFindSymbol("*", "webgui_start_browser_in_cef3");
   const char *cef_path = gSystem->Getenv("CEF_PATH");
   const char *rootsys = gSystem->Getenv("ROOTSYS");
   if (symbol_cef && cef_path && !gSystem->AccessPathName(cef_path) && rootsys && (is_native || ic_cef)) {
      typedef void (*FunctionCef3)(const char *, void *, bool, const char *, const char *);

      // addr.Form("/web7gui/%s/draw.htm?cef_canvas%s", GetName(), (IsBatchMode() ? "&batch_mode" : ""));
      addr.Form("/web7gui/%s/draw.htm?cef_canvas&no_root_json%s", handler->GetName(), (batch_mode ? "&batch_mode" : ""));

      printf("Show canvas in CEF window:  %s\n", addr.Data());

      FunctionCef3 func = (FunctionCef3)symbol_cef;
      func(addr.Data(), fServer, batch_mode, rootsys, cef_path);

      return true;
   }

   if (!CreateHttpServer(true)) {
      Error("NewDisplay", "Fail to start HTTP server");
      return false;
   }

   addr.Form("%s/web7gui/%s/draw.htm?webcanvas", fAddr.c_str(), handler->GetName());

   TString exec;

   if (!is_native && !ic_cef && !is_qt5 && (where != "browser")) {
      if (where.find("$url") != std::string::npos) {
         exec = where.c_str();
         exec.ReplaceAll("$url", addr);
      } else {
         exec.Form("%s %s", where.c_str(), addr.Data());
      }
   } else if (gSystem->InheritsFrom("TMacOSXSystem"))
      exec.Form("open %s", addr.Data());
   else
      exec.Form("xdg-open %s &", addr.Data());

   printf("Show canvas in browser with cmd:  %s\n", exec.Data());

   gSystem->Exec(exec);

   return true;
}

