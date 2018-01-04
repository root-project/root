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

#include "TCanvas.h"
#include "TSystem.h"
#include "TRandom.h"

ClassImp(TWebGuiFactory)


////////////////////////////////////////////////////////////////////////////////
/// TWebGuiFactory ctor.
/// Restore the right TVirtualX pointer

TWebGuiFactory::TWebGuiFactory() :
   TGuiFactory("WebRootProxy","web-based ROOT GUI Factory"),
   fGuiProxy(0)
{
   //if (TGQt::GetVirtualX())  gVirtualX = TGQt::GetVirtualX();
   // gSystem->Load("libGui");

   printf("Creating TWebGuiFactory\n");

   fGuiProxy = new TRootGuiFactory();

   if (!gVirtualX || gVirtualX->IsA() != TWebVirtualX::Class()) {
      TWebVirtualX *vx = new TWebVirtualX("webx", "redirection to native X", gVirtualX);
      gVirtualX = vx;
   }
}

////////////////////////////////////////////////////////////////////////////////

TWebGuiFactory::~TWebGuiFactory()
{
   delete fGuiProxy;
   fGuiProxy = 0;
}


////////////////////////////////////////////////////////////////////////////////

TApplicationImp *TWebGuiFactory::CreateApplicationImp(const char *classname, int *argc, char **argv)
{
   return fGuiProxy ? fGuiProxy->CreateApplicationImp(classname, argc, argv) : 0;
}


////////////////////////////////////////////////////////////////////////////////

TCanvasImp *TWebGuiFactory::CreateCanvasImp(TCanvas *c, const char *title, UInt_t width, UInt_t height)
{
   return new TWebCanvas(c, title, 0, 0, width, height);
}

////////////////////////////////////////////////////////////////////////////////

TCanvasImp *TWebGuiFactory::CreateCanvasImp(TCanvas *c, const char *title, Int_t x, Int_t y, UInt_t width, UInt_t height)
{
   return new TWebCanvas(c, title, x, y, width, height);
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
