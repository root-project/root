// Author: Sergey Linev, GSI   7/12/2016

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TWebGuiFactory                                                       //
//                                                                      //
// This class is a factory for Web GUI components. It overrides         //
// the member functions of the abstract TGuiFactory.                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TWebGuiFactory.h"
#include "TRootGuiFactory.h"

#include "TWebCanvas.h"

#include <ROOT/RMakeUnique.hxx>

////////////////////////////////////////////////////////////////////////////////
/// TWebGuiFactory ctor.
/// Restore the right TVirtualX pointer

TWebGuiFactory::TWebGuiFactory() :
   TGuiFactory("WebRootProxy","web-based ROOT GUI Factory")
{
   fGuiProxy = std::make_unique<TRootGuiFactory>();
}

////////////////////////////////////////////////////////////////////////////////

TApplicationImp *TWebGuiFactory::CreateApplicationImp(const char *classname, int *argc, char **argv)
{
   return fGuiProxy->CreateApplicationImp(classname, argc, argv);
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
   return fGuiProxy->CreateBrowserImp(b, title, width, height, opt);
}

////////////////////////////////////////////////////////////////////////////////

TBrowserImp *TWebGuiFactory::CreateBrowserImp(TBrowser *b, const char *title, Int_t x, Int_t y, UInt_t width, UInt_t height,Option_t *opt)
{
   return fGuiProxy->CreateBrowserImp(b, title, x, y, width, height, opt);
}

////////////////////////////////////////////////////////////////////////////////

TContextMenuImp *TWebGuiFactory::CreateContextMenuImp(TContextMenu *c, const char *name, const char *title)
{
   return fGuiProxy->CreateContextMenuImp(c, name, title);
}

////////////////////////////////////////////////////////////////////////////////

TControlBarImp *TWebGuiFactory::CreateControlBarImp(TControlBar *c, const char *title)
{
   return fGuiProxy->CreateControlBarImp(c,title);
}

////////////////////////////////////////////////////////////////////////////////

TControlBarImp *TWebGuiFactory::CreateControlBarImp(TControlBar *c, const char *title, Int_t x, Int_t y)
{
   return fGuiProxy->CreateControlBarImp(c, title, x, y);
}

////////////////////////////////////////////////////////////////////////////////

TInspectorImp *TWebGuiFactory::CreateInspectorImp(const TObject *obj, UInt_t width, UInt_t height)
{
   return fGuiProxy->CreateInspectorImp(obj, width, height);
}
