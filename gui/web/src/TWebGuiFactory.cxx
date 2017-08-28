// Author: Sergey Linev   7/12/2016
/****************************************************************************
**
** Copyright (C) 2016 by Sergey Linev.  All rights reserved.
**
*****************************************************************************/

//________________________________________________________________________
//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TWebGuiFactory                                                       //
//                                                                      //
// This class is a factory for Qt GUI components. It overrides          //
// the member functions of the ABS TGuiFactory.                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TWebGuiFactory.h"
#include "TRootGuiFactory.h"

#include "TWebCanvas.h"
#include "TWebVirtualX.h"

#include "THttpServer.h"
#include "THttpEngine.h"
#include "TCanvas.h"
#include "TSystem.h"
#include "TRandom.h"

ClassImp(TWebGuiFactory)


THttpServer * TWebGuiFactory::gServer = 0;

THttpServer *TWebGuiFactory::GetHttpServer() { return gServer; }

////////////////////////////////////////////////////////////////////////////////
/// TWebGuiFactory ctor.
/// Restore the right TVirtualX pointer

TWebGuiFactory::TWebGuiFactory() :
   TGuiFactory("WebRootProxy","web-based ROOT GUI Factory"),
   fGuiProxy(0),
   fAddr()
{
   //if (TGQt::GetVirtualX())  gVirtualX = TGQt::GetVirtualX();
   // gSystem->Load("libGui");

   printf("Creating TWebGuiFactory\n");

   fGuiProxy = new TRootGuiFactory();

   TWebVirtualX *vx = new TWebVirtualX("webx", "redirection to native X", gVirtualX);

   gVirtualX = vx;
}

////////////////////////////////////////////////////////////////////////////////

TWebGuiFactory::~TWebGuiFactory()
{
   if (gServer) {
      delete gServer;
      gServer = 0;
   }

   delete fGuiProxy;
   fGuiProxy = 0;
}


////////////////////////////////////////////////////////////////////////////////

TApplicationImp *TWebGuiFactory::CreateApplicationImp(const char *classname, int *argc, char **argv)
{

   return fGuiProxy ? fGuiProxy->CreateApplicationImp(classname, argc, argv) : 0;
}

////////////////////////////////////////////////////////////////////////////////

Bool_t TWebGuiFactory::CreateHttpServer()
{
   if (!gServer)
      gServer = new THttpServer("dummy");

   Bool_t with_http = !gSystem->DynFindSymbol("*", "webgui_start_browser_in_qt5") &&
                      !gSystem->DynFindSymbol("*", "webgui_start_browser_in_cef3");

   if (!with_http || (fAddr.Length() > 0)) return kTRUE;

   // gServer = new THttpServer("http:8080?loopback&websocket_timeout=10000");

   int http_port = 0;
   const char *ports = gSystem->Getenv("WEBGUI_PORT");
   if (ports)
      http_port = TString(ports).Atoi();
   if (!http_port)
      gRandom->SetSeed(0);

   for (int ntry = 0; ntry < 100; ++ntry) {
      if (!http_port)
         http_port = (int)(8800 + 1000 * gRandom->Rndm(1));

      if (gServer->CreateEngine(TString::Format("http:%d?websocket_timeout=10000&thrds=20", http_port))) {
         fAddr.Form("http://localhost:%d", http_port);
         return kTRUE;
      }

      http_port = 0;
   }

   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////

TCanvasImp *TWebGuiFactory::CreateCanvasImp(TCanvas *c, const char *title, UInt_t width, UInt_t height)
{
   CreateHttpServer();

   TWebCanvas *res = new TWebCanvas(c, title, 0, 0, width, height, fAddr, gServer);

   gServer->Register("/web6gui", res);

   return res;
}

////////////////////////////////////////////////////////////////////////////////

TCanvasImp *TWebGuiFactory::CreateCanvasImp(TCanvas *c, const char *title, Int_t x, Int_t y, UInt_t width, UInt_t height)
{
   CreateHttpServer();

   TWebCanvas *res = new TWebCanvas(c, title, x, y, width, height, fAddr, gServer);

   gServer->Register("/web6gui", res);

   return res;
}

////////////////////////////////////////////////////////////////////////////////

TBrowserImp *TWebGuiFactory::CreateBrowserImp(TBrowser *b, const char *title, UInt_t width, UInt_t height)
{
   return CreateBrowserImp(b, title, width, height, (Option_t *)0);
}

////////////////////////////////////////////////////////////////////////////////

TBrowserImp *TWebGuiFactory::CreateBrowserImp(TBrowser *b, const char *title, Int_t x, Int_t y, UInt_t width, UInt_t height)
{
  return CreateBrowserImp(b, title, x, y, width, height, (Option_t *)0);
}

////////////////////////////////////////////////////////////////////////////////

TBrowserImp *TWebGuiFactory::CreateBrowserImp(TBrowser *b, const char *title, UInt_t width, UInt_t height, Option_t *opt)
{
   return fGuiProxy ? fGuiProxy->CreateBrowserImp(b, title, width, height, opt) : 0;
}

////////////////////////////////////////////////////////////////////////////////

TBrowserImp *TWebGuiFactory::CreateBrowserImp(TBrowser *b, const char *title, Int_t x, Int_t y, UInt_t width, UInt_t height,Option_t *opt)
{
   return fGuiProxy ? fGuiProxy->CreateBrowserImp(b, title, x, y, width, height, opt) : 0;
}

////////////////////////////////////////////////////////////////////////////////

TContextMenuImp *TWebGuiFactory::CreateContextMenuImp(TContextMenu *c, const char *name, const char *title)
{
   return fGuiProxy ? fGuiProxy->CreateContextMenuImp(c, name, title): 0;
}

////////////////////////////////////////////////////////////////////////////////

TControlBarImp *TWebGuiFactory::CreateControlBarImp(TControlBar *c, const char *title)
{
   return fGuiProxy ? fGuiProxy->CreateControlBarImp(c,title) : 0;
}

////////////////////////////////////////////////////////////////////////////////

TControlBarImp *TWebGuiFactory::CreateControlBarImp(TControlBar *c, const char *title, Int_t x, Int_t y)
{
   return fGuiProxy ? fGuiProxy->CreateControlBarImp(c, title, x, y):0;
}

////////////////////////////////////////////////////////////////////////////////

TInspectorImp *TWebGuiFactory::CreateInspectorImp(const TObject *obj, UInt_t width, UInt_t height)
{
   return fGuiProxy ? fGuiProxy->CreateInspectorImp(obj, width, height) :0 ;
}
