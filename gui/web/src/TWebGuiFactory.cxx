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
   fAddr("http://localhost:8181")
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

void TWebGuiFactory::CreateHttpServer()
{
   if (gServer) return;

   if (gSystem->DynFindSymbol("*", "webgui_start_browser_new")) {
      gServer = new THttpServer("nope"); // server without any external engine
   } else {
      // gServer = new THttpServer("http:8080?loopback&websocket_timeout=10000");
     const char *port = gSystem->Getenv("WEBGUI_PORT");
     TString buf;
     if (!port) {
        gRandom->SetSeed(0);
        buf.Form("%d", (int) (8800 + 1000* gRandom->Rndm(1)));
        port = buf.Data(); // "8181";
     }
     fAddr.Form("http://localhost:%s", port);
     gServer = new THttpServer(Form("http:%s?websocket_timeout=10000", port));
   }
}

////////////////////////////////////////////////////////////////////////////////

TCanvasImp *TWebGuiFactory::CreateCanvasImp(TCanvas *c, const char *title, UInt_t width, UInt_t height)
{
   CreateHttpServer();

   TWebCanvas *res = new TWebCanvas(c, title, 0, 0, width, height, fAddr);

   gServer->Register("/webgui", res);

   return res;
}

////////////////////////////////////////////////////////////////////////////////

TCanvasImp *TWebGuiFactory::CreateCanvasImp(TCanvas *c, const char *title, Int_t x, Int_t y, UInt_t width, UInt_t height)
{
   CreateHttpServer();

   TWebCanvas *res = new TWebCanvas(c, title, x, y, width, height, fAddr);

   gServer->Register("/webgui", res);

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
