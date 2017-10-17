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

#include "THttpServer.h"
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
      fServer->Unregister((TNamed *) display->fWSHandler);

   for (auto displ = fDisplays.begin(); displ != fDisplays.end(); displ++) {
      if (displ->get() == display) {
         fDisplays.erase(displ);
         break;
      }
   }

}



