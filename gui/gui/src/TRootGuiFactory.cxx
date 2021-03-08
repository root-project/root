// @(#)root/gui:$Id$
// Author: Fons Rademakers   15/01/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


/** \class TRootGuiFactory
    \ingroup guiwidgets

This class is a factory for ROOT GUI components. It overrides
the member functions of the ABS TGuiFactory.

*/


#include "TRootGuiFactory.h"
#include "TRootApplication.h"
#include "TRootCanvas.h"
#include "TRootBrowser.h"
#include "TRootBrowserLite.h"
#include "TRootContextMenu.h"
#include "TRootControlBar.h"
#include "TROOT.h"
#include "TPluginManager.h"
#include "TEnv.h"

ClassImp(TRootGuiFactory);

////////////////////////////////////////////////////////////////////////////////
/// TRootGuiFactory ctor.

TRootGuiFactory::TRootGuiFactory(const char *name, const char *title)
   : TGuiFactory(name, title)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Create a ROOT native GUI version of TApplicationImp

TApplicationImp *TRootGuiFactory::CreateApplicationImp(const char *classname,
                      Int_t *argc, char **argv)
{
   TRootApplication *app = new TRootApplication(classname, argc, argv);
   if (!app->Client()) {
      delete app;
      app = 0;
   }
   return app;
}

////////////////////////////////////////////////////////////////////////////////
/// Create a ROOT native GUI version of TCanvasImp

TCanvasImp *TRootGuiFactory::CreateCanvasImp(TCanvas *c, const char *title,
                                             UInt_t width, UInt_t height)
{
   return new TRootCanvas(c, title, width, height);
}

////////////////////////////////////////////////////////////////////////////////
/// Create a ROOT native GUI version of TCanvasImp

TCanvasImp *TRootGuiFactory::CreateCanvasImp(TCanvas *c, const char *title,
                                  Int_t x, Int_t y, UInt_t width, UInt_t height)
{
   return new TRootCanvas(c, title, x, y, width, height);
}

////////////////////////////////////////////////////////////////////////////////
/// Create a ROOT native GUI version of TBrowserImp

TBrowserImp *TRootGuiFactory::CreateBrowserImp(TBrowser *b, const char *title,
                                               UInt_t width, UInt_t height,
                                               Option_t *opt)
{
   TString browserVersion(gEnv->GetValue("Browser.Name", "TRootBrowserLite"));
   TPluginHandler *ph = gROOT->GetPluginManager()->FindHandler("TBrowserImp",
                                                               browserVersion);
   TString browserOptions(gEnv->GetValue("Browser.Options", "FECI"));
   if (opt && strlen(opt))
      browserOptions = opt;
   browserOptions.ToUpper();
   if (browserOptions.Contains("LITE"))
      return new TRootBrowserLite(b, title, width, height);
   if (ph && ph->LoadPlugin() != -1) {
      TBrowserImp *imp = (TBrowserImp *)ph->ExecPlugin(5, b, title, width,
         height, browserOptions.Data());
      if (imp) return imp;
   }
   if (browserVersion.Contains("Lite", TString::kIgnoreCase))
      return new TRootBrowserLite(b, title, width, height);
   else
      return new TRootBrowser(b, title, width, height, browserOptions.Data());
}

////////////////////////////////////////////////////////////////////////////////
/// Create a ROOT native GUI version of TBrowserImp

TBrowserImp *TRootGuiFactory::CreateBrowserImp(TBrowser *b, const char *title,
                                               Int_t x, Int_t y, UInt_t width,
                                               UInt_t height, Option_t *opt)
{
   TString browserVersion(gEnv->GetValue("Browser.Name", "TRootBrowserLite"));
   TPluginHandler *ph = gROOT->GetPluginManager()->FindHandler("TBrowserImp",
                                                               browserVersion);
   TString browserOptions(gEnv->GetValue("Browser.Options", "FECI"));
   if (opt && strlen(opt))
      browserOptions = opt;
   browserOptions.ToUpper();
   if (browserOptions.Contains("LITE"))
      return new TRootBrowserLite(b, title, width, height);
   if (ph && ph->LoadPlugin() != -1) {
      TBrowserImp *imp = (TBrowserImp *)ph->ExecPlugin(7, b, title, x, y, width,
         height, browserOptions.Data());
      if (imp) return imp;
   }
   if (browserVersion.Contains("Lite", TString::kIgnoreCase))
      return new TRootBrowserLite(b, title, x, y, width, height);
   else
      return new TRootBrowser(b, title, x, y, width, height, browserOptions.Data());
}

////////////////////////////////////////////////////////////////////////////////
/// Create a ROOT native GUI version of TContextMenuImp

TContextMenuImp *TRootGuiFactory::CreateContextMenuImp(TContextMenu *c,
                                             const char *name, const char *)
{
   return new TRootContextMenu(c, name);
}

////////////////////////////////////////////////////////////////////////////////
/// Create a ROOT native GUI version of TControlBarImp

TControlBarImp *TRootGuiFactory::CreateControlBarImp(TControlBar *c, const char *title)
{
   return new TRootControlBar(c, title);
}

////////////////////////////////////////////////////////////////////////////////
/// Create a ROOT native GUI version of TControlBarImp

TControlBarImp *TRootGuiFactory::CreateControlBarImp(TControlBar *c, const char *title,
                                                     Int_t x, Int_t y)
{
   return new TRootControlBar(c, title, x, y);
}
