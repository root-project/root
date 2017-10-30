// Copyright (c) 2013 The Chromium Embedded Framework Authors. All rights
// reserved. Use of this source code is governed by a BSD-style license that
// can be found in the LICENSE file.

#ifndef CEF_TESTS_CEFSIMPLE_SIMPLE_APP_H_
#define CEF_TESTS_CEFSIMPLE_SIMPLE_APP_H_

#include "include/cef_app.h"

#include "gui_handler.h"
#include "osr_handler.h"

class THttpServer;

// Implement application-level callbacks for the browser process.
class SimpleApp : public CefApp, public CefBrowserProcessHandler /*, public CefRenderProcessHandler */ {
protected:
   std::string fUrl;     ///<! first URL to open
   std::string fCefMain; ///!< executable used for extra processed
   bool fBatch;          ///!< indicate batch mode
   CefRect fRect;        ///!< original width

   CefRefPtr<OsrHandler> fOsrHandler; ///!< batch-mode handler
   bool fUseViewes;                   ///!< is viewes are used
   CefRefPtr<GuiHandler> fGuiHandler; ///!< normal handler

public:
   SimpleApp(const std::string &url, const std::string &cef_main, THttpServer *server = 0, bool isbatch = false);
   virtual ~SimpleApp();

   void SetRect(unsigned width, unsigned height) { fRect.Set(0, 0, width, height); }

   // CefApp methods:
   virtual CefRefPtr<CefBrowserProcessHandler> GetBrowserProcessHandler() OVERRIDE { return this; }
   // virtual CefRefPtr<CefRenderProcessHandler> GetRenderProcessHandler() OVERRIDE { return this; }

   virtual void OnRegisterCustomSchemes(CefRawPtr<CefSchemeRegistrar> registrar) OVERRIDE;

   // CefBrowserProcessHandler methods:
   virtual void OnContextInitialized() OVERRIDE;

   virtual void
   OnBeforeCommandLineProcessing(const CefString &process_type, CefRefPtr<CefCommandLine> command_line) OVERRIDE;

   virtual void OnBeforeChildProcessLaunch(CefRefPtr<CefCommandLine> command_line) OVERRIDE;

   void StartWindow(const std::string &url, bool batch, CefRect &rect);

   // CefRenderProcessHandler methods
   // virtual void OnContextCreated(CefRefPtr<CefBrowser> browser, CefRefPtr<CefFrame> frame,
   //                              CefRefPtr<CefV8Context> context) OVERRIDE;

private:
   // Include the default reference counting implementation.
   IMPLEMENT_REFCOUNTING(SimpleApp);
   DISALLOW_COPY_AND_ASSIGN(SimpleApp);
};

#endif // CEF_TESTS_CEFSIMPLE_SIMPLE_APP_H_
