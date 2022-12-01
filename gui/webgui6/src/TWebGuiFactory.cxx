// Author: Sergey Linev, GSI   7/12/2016

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TWebGuiFactory.h"

#include "TWebCanvas.h"
#include "TEnv.h"
#include "TROOT.h"
#include "TPluginManager.h"

/** \class TWebGuiFactory
\ingroup webgui6

This class is a proxy-factory for web-base ROOT GUI components.
It allows to create canvas and browser implementations.

*/

////////////////////////////////////////////////////////////////////////////////
/// TWebGuiFactory ctor.

TWebGuiFactory::TWebGuiFactory() :
   TGuiFactory("WebRootProxy", "web-based ROOT GUI Factory")
{
}

////////////////////////////////////////////////////////////////////////////////

TCanvasImp *TWebGuiFactory::CreateCanvasImp(TCanvas *c, const char *title, UInt_t width, UInt_t height)
{
   Bool_t readonly = gEnv->GetValue("WebGui.FullCanvas", (Int_t) 1) == 0;

   return new TWebCanvas(c, title, 0, 0, width, height, readonly);
}

////////////////////////////////////////////////////////////////////////////////

TCanvasImp *TWebGuiFactory::CreateCanvasImp(TCanvas *c, const char *title, Int_t x, Int_t y, UInt_t width, UInt_t height)
{
   Bool_t readonly = gEnv->GetValue("WebGui.FullCanvas", (Int_t) 1) == 0;

   return new TWebCanvas(c, title, x, y, width, height, readonly);
}

////////////////////////////////////////////////////////////////////////////////

TBrowserImp *TWebGuiFactory::CreateBrowserImp(TBrowser *b, const char *title, UInt_t width, UInt_t height, Option_t *opt)
{
   auto ph = gROOT->GetPluginManager()->FindHandler("TBrowserImp", "ROOT::Experimental::RWebBrowserImp");

   if (ph && (ph->LoadPlugin() != -1)) {
      TBrowserImp *imp = (TBrowserImp *)ph->ExecPlugin(5, b, title, width, height, opt);
      if (imp) return imp;
   }

   return TGuiFactory::CreateBrowserImp(b, title, width, height, opt);
}

////////////////////////////////////////////////////////////////////////////////

TBrowserImp *TWebGuiFactory::CreateBrowserImp(TBrowser *b, const char *title, Int_t x, Int_t y, UInt_t width, UInt_t height, Option_t *opt)
{
   auto ph = gROOT->GetPluginManager()->FindHandler("TBrowserImp", "ROOT::Experimental::RWebBrowserImp");

   if (ph && (ph->LoadPlugin() != -1)) {
      TBrowserImp *imp = (TBrowserImp *)ph->ExecPlugin(7, b, title, x, y, width, height, opt);
      if (imp) return imp;
   }

   return TGuiFactory::CreateBrowserImp(b, title, x, y, width, height, opt);
}
