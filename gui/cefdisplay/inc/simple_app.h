// Author: Sergey Linev <S.Linev@gsi.de>
// Date: 2017-06-29
// Warning: This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!

// Copyright (c) 2013 The Chromium Embedded Framework Authors. All rights
// reserved. Use of this source code is governed by a BSD-style license that
// can be found in the LICENSE file.

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_cef_simple_app
#define ROOT_cef_simple_app

#include "include/cef_app.h"

#include <string>

#include "gui_handler.h"

class THttpServer;

class RCefWebDisplayHandle;

// Implement application-level callbacks for the browser process.
class SimpleApp : public CefApp,
#if defined(OS_LINUX)
                  public CefPrintHandler,
#endif
                  /*, public CefRenderProcessHandler */
                  public CefBrowserProcessHandler {
protected:
   bool fUseViewes{false};  ///<! is views framework used
   THttpServer *fFirstServer; ///<! first server
   std::string fFirstUrl;   ///<! first URL to open
   std::string fFirstContent; ///<! first page content open
   CefRect fFirstRect;      ///<! original width
   bool fFirstHeadless{false}; ///<! is first window is headless
   RCefWebDisplayHandle *fNextHandle{nullptr}; ///< next handle where browser will be created

   CefRefPtr<GuiHandler> fGuiHandler; ///<! normal handler

public:
   SimpleApp(bool use_viewes,
             THttpServer *serv = nullptr, const std::string &url = "", const std::string &cont = "",
             int width = 0, int height = 0, bool headless = false);

   void SetNextHandle(RCefWebDisplayHandle *handle);

   // CefApp methods:
   virtual CefRefPtr<CefBrowserProcessHandler> GetBrowserProcessHandler() override { return this; }

#if defined(OS_LINUX)
#if CEF_VERSION_MAJOR < 95
   // only on Linux special print handler is required to return PDF size
   virtual CefRefPtr<CefPrintHandler> GetPrintHandler() override { return this; }
#endif
#endif
   // virtual CefRefPtr<CefRenderProcessHandler> GetRenderProcessHandler() override { return this; }

   virtual void OnRegisterCustomSchemes(CefRawPtr<CefSchemeRegistrar> registrar) override;

   // CefBrowserProcessHandler methods:
   virtual void OnContextInitialized() override;

   virtual void
   OnBeforeCommandLineProcessing(const CefString &process_type, CefRefPtr<CefCommandLine> command_line) override;

   virtual void OnBeforeChildProcessLaunch(CefRefPtr<CefCommandLine> command_line) override;

#if defined(OS_LINUX)
   // CefPrintHandler methods
#if CEF_VERSION_MAJOR < 95
   virtual CefSize GetPdfPaperSize(int device_units_per_inch) override { return CefSize(device_units_per_inch*8.25, device_units_per_inch*11.75); }
#else
   virtual CefSize GetPdfPaperSize(CefRefPtr<CefBrowser>, int device_units_per_inch) override { return CefSize(device_units_per_inch*8.25, device_units_per_inch*11.75); }
#endif
   virtual bool OnPrintDialog( CefRefPtr< CefBrowser > browser, bool has_selection, CefRefPtr< CefPrintDialogCallback > callback ) override { return false; }
   virtual bool OnPrintJob( CefRefPtr< CefBrowser > browser, const CefString& document_name, const CefString& pdf_file_path, CefRefPtr< CefPrintJobCallback > callback ) override { return false; }
   virtual void OnPrintReset( CefRefPtr< CefBrowser > browser ) override {}
   virtual void OnPrintSettings( CefRefPtr< CefBrowser > browser, CefRefPtr< CefPrintSettings > settings, bool get_defaults ) override {}
   virtual void OnPrintStart( CefRefPtr< CefBrowser > browser ) override {}
#endif


   void StartWindow(THttpServer *serv, const std::string &url, const std::string &cont, CefRect &rect);

   // CefRenderProcessHandler methods
   // virtual void OnContextCreated(CefRefPtr<CefBrowser> browser, CefRefPtr<CefFrame> frame,
   //                              CefRefPtr<CefV8Context> context) override;

private:
   // Include the default reference counting implementation.
   IMPLEMENT_REFCOUNTING(SimpleApp);
   DISALLOW_COPY_AND_ASSIGN(SimpleApp);
};

#endif // ROOT_cef_simple_app
