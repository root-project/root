// @(#)root/base:$Id$
// Author: Fons Rademakers   15/11/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TGuiFactory
\ingroup Base

This ABC is a factory for GUI components. Depending on which
factory is active one gets either ROOT native (X11 based with Win95
look and feel), Win32 or Mac components.

In case there is no platform dependent implementation on can run in
batch mode directly using an instance of this base class.
*/

#include "TGuiFactory.h"
#include "TApplicationImp.h"
#include "TCanvasImp.h"
#include "TBrowserImp.h"
#include "TContextMenuImp.h"
#include "TControlBarImp.h"
#include "TInspectorImp.h"
#include "TROOT.h"
#include "TEnv.h"
#include "TPluginManager.h"

TGuiFactory *gGuiFactory = nullptr;
TGuiFactory *gBatchGuiFactory = nullptr;

ClassImp(TGuiFactory);

////////////////////////////////////////////////////////////////////////////////
/// TGuiFactory ctor only called by derived classes.

TGuiFactory::TGuiFactory(const char *name, const char *title)
    : TNamed(name, title)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Create a batch version of TApplicationImp.

TApplicationImp *TGuiFactory::CreateApplicationImp(const char *classname, int *argc, char **argv)
{
   return new TApplicationImp(classname, argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
/// Create a batch version of TCanvasImp.

TCanvasImp *TGuiFactory::CreateCanvasImp(TCanvas *c, const char *title, UInt_t width, UInt_t height)
{
   if (gROOT->IsWebDisplay()) {
      auto ph = gROOT->GetPluginManager()->FindHandler("TCanvasImp", "TWebCanvas");

      if (ph && ph->LoadPlugin() != -1) {
         auto imp = (TCanvasImp *)ph->ExecPlugin(6, c, title, 0, 0, width, height);
         if (imp)
            return imp;
      }

      Error("CreateCanvasImp", "Fail to create TWebCanvas, please provide missing libWebGui6 or run 'root --web=off'");
   }

   return new TCanvasImp(c, title, width, height);
}

////////////////////////////////////////////////////////////////////////////////
/// Create a batch version of TCanvasImp.

TCanvasImp *TGuiFactory::CreateCanvasImp(TCanvas *c, const char *title, Int_t x, Int_t y, UInt_t width, UInt_t height)
{
   if (gROOT->IsWebDisplay()) {
      auto ph = gROOT->GetPluginManager()->FindHandler("TCanvasImp", "TWebCanvas");

      if (ph && ph->LoadPlugin() != -1) {
         auto imp = (TCanvasImp *)ph->ExecPlugin(6, c, title, x, y, width, height);
         if (imp)
            return imp;
      }

      Error("CreateCanvasImp", "Fail to create TWebCanvas, please provide missing libWebGui6 or run 'root --web=off'");
   }

   return new TCanvasImp(c, title, x, y, width, height);
}

////////////////////////////////////////////////////////////////////////////////
/// Create a batch version of TBrowserImp.

TBrowserImp *TGuiFactory::CreateBrowserImp(TBrowser *b, const char *title, UInt_t width, UInt_t height, Option_t *opt)
{
   const char *browserName = nullptr;

   if (gROOT->IsWebDisplay() && !gROOT->IsWebDisplayBatch())
      browserName = "ROOT::RWebBrowserImp";
   else if (!gROOT->IsBatch())
      browserName = gEnv->GetValue("Browser.Name", "");

   if (browserName && *browserName) {
      auto ph = gROOT->GetPluginManager()->FindHandler("TBrowserImp", browserName);

      if (ph && ph->LoadPlugin() != -1) {
         auto imp = (TBrowserImp *)ph->ExecPlugin(5, b, title, width, height, opt);
         if (imp)
            return imp;
      }

      Error("CreateBrowserImp", "Fail to create %s, please provide missing libraries or run 'root --web=off'",
            browserName);
   }

   return new TBrowserImp(b, title, width, height);
}

////////////////////////////////////////////////////////////////////////////////
/// Create a batch version of TBrowserImp.

TBrowserImp *TGuiFactory::CreateBrowserImp(TBrowser *b, const char *title, Int_t x, Int_t y, UInt_t width, UInt_t height, Option_t *opt)
{
   const char *browserName = nullptr;

   if (gROOT->IsWebDisplay() && !gROOT->IsWebDisplayBatch())
      browserName = "ROOT::RWebBrowserImp";
   else if (!gROOT->IsBatch())
      browserName = gEnv->GetValue("Browser.Name", "");

   if (browserName && *browserName) {
      auto ph = gROOT->GetPluginManager()->FindHandler("TBrowserImp", browserName);

      if (ph && ph->LoadPlugin() != -1) {
         auto imp = (TBrowserImp *)ph->ExecPlugin(7, b, title, x, y, width, height, opt);
         if (imp)
            return imp;
      }

      Error("CreateBrowserImp", "Fail to create %s, please provide missing libraries or run 'root --web=off'",
            browserName);
   }

   return new TBrowserImp(b, title, x, y, width, height);
}

////////////////////////////////////////////////////////////////////////////////
/// Create a batch version of TContextMenuImp.

TContextMenuImp *TGuiFactory::CreateContextMenuImp(TContextMenu *c, const char *, const char *)
{
   return new TContextMenuImp(c);
}

////////////////////////////////////////////////////////////////////////////////
/// Create a batch version of TControlBarImp.

TControlBarImp *TGuiFactory::CreateControlBarImp(TControlBar *c, const char *title)
{
   if (gROOT->IsWebDisplay()) {
      auto ph = gROOT->GetPluginManager()->FindHandler("TControlBarImp", "TWebControlBar");

      if (ph && ph->LoadPlugin() != -1) {
         auto imp = (TControlBarImp *)ph->ExecPlugin(4, c, title, 0, 0);
         if (imp)
            return imp;
      }

      Error("CreateControlBarImp",
            "Fail to create TWebControlBar, please provide missing libWebGui6 or run 'root --web=off'");
   }

   return new TControlBarImp(c, title);
}

////////////////////////////////////////////////////////////////////////////////
/// Create a batch version of TControlBarImp.

TControlBarImp *TGuiFactory::CreateControlBarImp(TControlBar *c, const char *title, Int_t x, Int_t y)
{
   if (gROOT->IsWebDisplay()) {
      auto ph = gROOT->GetPluginManager()->FindHandler("TControlBarImp", "TWebControlBar");

      if (ph && ph->LoadPlugin() != -1) {
         auto imp = (TControlBarImp *)ph->ExecPlugin(4, c, title, x, y);
         if (imp)
            return imp;
      }

      Error("CreateControlBarImp",
            "Fail to create TWebControlBar, please provide missing libWebGui6 or run 'root --web=off'");
   }

   return new TControlBarImp(c, title, x, y);
}

////////////////////////////////////////////////////////////////////////////////
/// Create a batch version of TInspectorImp.

TInspectorImp *TGuiFactory::CreateInspectorImp(const TObject *obj, UInt_t width, UInt_t height)
{
   if (gROOT->IsBatch())
      return new TInspectorImp(obj, width, height);

   gROOT->ProcessLine(TString::Format("TInspectCanvas::Inspector((TObject*)0x%zx);", (size_t)obj).Data());
   return nullptr;
}
