// Author: Sergey Linev <S.Linev@gsi.de>
// Date: 2020-08-21
// Warning: This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!

/*************************************************************************
 * Copyright (C) 1995-2023, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#if !defined(_MSC_VER)
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wshadow"
#endif


#include "RCefWebDisplayHandle.hxx"

#include "TTimer.h"
#include "TROOT.h"
#include "TSystem.h"
#include "TEnv.h"
#include "TApplication.h"

#include "include/base/cef_build.h"

#include <ROOT/RLogger.hxx>

#include <memory>


class TCefTimer : public TTimer {
public:
   TCefTimer(Long_t milliSec, Bool_t mode) : TTimer(milliSec, mode) {}
   void Timeout() override
   {
      // just let run loop
      CefDoMessageLoopWork();
   }
};


class FrameSourceVisitor : public CefStringVisitor {

   RCefWebDisplayHandle *fHandle{nullptr};

public:

   FrameSourceVisitor(RCefWebDisplayHandle *handle) : CefStringVisitor(), fHandle(handle) {}

   ~FrameSourceVisitor() override = default;

   void Visit(const CefString &str) override
   {
      if (fHandle && fHandle->IsValid())
         fHandle->SetContent(str.ToString());
   }

private:
   // Include the default reference counting implementation.
   IMPLEMENT_REFCOUNTING(FrameSourceVisitor);
   DISALLOW_COPY_AND_ASSIGN(FrameSourceVisitor);
};


class HeadlessPrintCallback : public CefPdfPrintCallback {
   bool *fFlag{nullptr};
public:
   HeadlessPrintCallback(bool *flag) : CefPdfPrintCallback(), fFlag(flag) {}
   ~HeadlessPrintCallback() override = default;

   void OnPdfPrintFinished(const CefString&, bool ok) override
   {
      if (fFlag) *fFlag = true;
   }
private:
   // Include the default reference counting implementation.
   IMPLEMENT_REFCOUNTING(HeadlessPrintCallback);
   DISALLOW_COPY_AND_ASSIGN(HeadlessPrintCallback);
};

std::unique_ptr<ROOT::RWebDisplayHandle> RCefWebDisplayHandle::CefCreator::Display(const ROOT::RWebDisplayArgs &args)
{

   auto handle = std::make_unique<RCefWebDisplayHandle>(args.GetFullUrl());
   if (!args.IsStandalone())
      handle->fCloseBrowser = false;

   Int_t wait_tmout = args.IsHeadless() ? gEnv->GetValue("WebGui.CefHeadlessTimeout", 30) : -1;

   if (fCefApp) {
      fCefApp->SetNextHandle(handle.get(), args.IsHeadless());

      CefRect rect((args.GetX() > 0) ? args.GetX() : 0, (args.GetY() > 0) ? args.GetY() : 0,
            (args.GetWidth() > 0) ? args.GetWidth() : 800, (args.GetHeight() > 0) ? args.GetHeight() : 600);

      fCefApp->StartWindow(args.GetHttpServer(), args.GetFullUrl(), args.GetPageContent(), rect);

      if (args.IsHeadless())
         handle->WaitForContent(wait_tmout, args.GetExtraArgs());

      return handle;
   }

   GuiHandler::PlatformInit();
   bool use_views = true;

   TString env_use_views = gEnv->GetValue("WebGui.CefUseViews", "");
   if ((env_use_views == "yes") || (env_use_views == "1"))
      use_views = true;
   else if ((env_use_views == "no") || (env_use_views == "0"))
      use_views = false;

   // Specify CEF global settings here.
   CefSettings settings;

   TString ceflog = gEnv->GetValue("WebGui.CefLogSeveriry", "fatal");
   if (ceflog == "fatal")
      settings.log_severity = LOGSEVERITY_FATAL;
   else if (ceflog == "verbose")
      settings.log_severity = LOGSEVERITY_VERBOSE;
   else if (ceflog == "info")
      settings.log_severity = LOGSEVERITY_INFO;
   else if (ceflog == "warning")
      settings.log_severity = LOGSEVERITY_WARNING;
   else if (ceflog == "error")
      settings.log_severity = LOGSEVERITY_ERROR;
   else if (ceflog == "disable")
      settings.log_severity = LOGSEVERITY_DISABLE;
   else
      settings.log_severity = LOGSEVERITY_FATAL;

   bool supress_log = (settings.log_severity == LOGSEVERITY_DISABLE) ||
                      (settings.log_severity == LOGSEVERITY_FATAL);

   TApplication *root_app = gROOT->GetApplication();

   std::vector<const char *> cef_argv = { root_app->Argv(0) };

#ifdef OS_WIN

   CefMainArgs main_args(args.IsHeadless() ? (HINSTANCE) 0 : GetModuleHandle(nullptr));

#else

   if (args.IsHeadless()) {
      cef_argv.emplace_back("--user-data-dir=.");
      cef_argv.emplace_back("--allow-file-access-from-files");
      cef_argv.emplace_back("--disable-web-security");
#ifdef OS_LINUX
      cef_argv.emplace_back("--disable-gpu");
      cef_argv.emplace_back("--ignore-gpu-blocklist");
      cef_argv.emplace_back("--use-gl=swiftshader");
      cef_argv.emplace_back("--enable-unsafe-swiftshader");
#endif
#ifdef OS_MACOSX
      cef_argv.emplace_back("--use-angle=metal");
      cef_argv.emplace_back("--ignore-gpu-blocklist");
      cef_argv.emplace_back("--enable-webgl");
      cef_argv.emplace_back("--enable-gpu");
      cef_argv.emplace_back("--enable-gpu-rasterization");
#endif
      cef_argv.emplace_back("--off-screen-rendering-enabled");
      if (use_views)
         cef_argv.emplace_back("--ozone-platform=headless");
   } else {
#ifdef OS_MACOSX
      cef_argv.emplace_back("--use-angle=metal");
      cef_argv.emplace_back("--ignore-gpu-blocklist");
      cef_argv.emplace_back("--enable-webgl");
#endif
   }

   if (supress_log) {
      cef_argv.emplace_back("--disable-logging");
      cef_argv.emplace_back("--enable-logging=none");
      cef_argv.emplace_back("--v=-1");
   }

   cef_argv.emplace_back(nullptr);

   CefMainArgs main_args(cef_argv.size() - 1, (char **) cef_argv.data());

#endif

   // CEF applications have multiple sub-processes (render, plugin, GPU, etc)
   // that share the same executable. This function checks the command-line and,
   // if this is a sub-process, executes the appropriate logic.

   /*         int exit_code = CefExecuteProcess(main_args, nullptr, nullptr);
        if (exit_code >= 0) {
          // The sub-process has completed so return here.
          return exit_code;
        }
    */

   // Install xlib error handlers so that the application won't be terminated
   // on non-fatal errors.
   //         XSetErrorHandler(XErrorHandlerImpl);
   //         XSetIOErrorHandler(XIOErrorHandlerImpl);


   TString cef_main = TROOT::GetBinDir() + "/cef_main";
   cef_string_ascii_to_utf16(cef_main.Data(), cef_main.Length(), &settings.browser_subprocess_path);

#ifdef OS_MACOSX
   // on mac there is framework directory, where resources and libs are combined together
   TString path = TROOT::GetDataDir() + "/Frameworks/Chromium Embedded Framework.framework";
   cef_string_ascii_to_utf16(path.Data(), path.Length(), &settings.framework_dir_path);

   // add CEF libraries to DYLD library path
   TString dypath = gSystem->Getenv("DYLD_LIBRARY_PATH");
   if (dypath.Length() > 0)
      dypath.Append(":");
   dypath.Append(path + "/Libraries/");
   gSystem->Setenv("DYLD_LIBRARY_PATH", dypath);
#endif

#ifdef OS_WIN
   TString resource_dir = TROOT::GetBinDir();
   cef_string_ascii_to_utf16(resource_dir.Data(), resource_dir.Length(), &settings.resources_dir_path);
   TString locales_dir = TROOT::GetDataDir() + "/Frameworks/cef/locales";
   cef_string_ascii_to_utf16(locales_dir.Data(), locales_dir.Length(), &settings.locales_dir_path);
#endif

#ifdef OS_LINUX
   TString resource_dir = TROOT::GetLibDir();
   cef_string_ascii_to_utf16(resource_dir.Data(), resource_dir.Length(), &settings.resources_dir_path);
   TString locales_dir = TROOT::GetDataDir() + "/Frameworks/cef/locales";
   cef_string_ascii_to_utf16(locales_dir.Data(), locales_dir.Length(), &settings.locales_dir_path);
#endif

   settings.no_sandbox = true;
   // if (gROOT->IsWebDisplayBatch()) settings.single_process = true;

   if (args.IsHeadless())
      settings.windowless_rendering_enabled = true;

   // settings.external_message_pump = true;
   // settings.multi_threaded_message_loop = false;

   std::string plog = "cef.log";
   cef_string_ascii_to_utf16(plog.c_str(), plog.length(), &settings.log_file);

   // settings.uncaught_exception_stack_size = 100;
   // settings.ignore_certificate_errors = true;

   // settings.remote_debugging_port = 7890;

   // SimpleApp implements application-level callbacks for the browser process.
   // It will create the first browser instance in OnContextInitialized() after
   // CEF has initialized.
   fCefApp = new SimpleApp(use_views, supress_log,
                           args.GetHttpServer(), args.GetFullUrl(), args.GetPageContent(),
                           args.GetWidth() > 0 ? args.GetWidth() : 800,
                           args.GetHeight() > 0 ? args.GetHeight() : 600,
                           args.IsHeadless());

   fCefApp->SetNextHandle(handle.get(), args.IsHeadless());

   // Initialize CEF for the browser process.
   CefInitialize(main_args, settings, fCefApp.get(), nullptr);

   if (args.IsHeadless()) {
      handle->WaitForContent(wait_tmout, args.GetExtraArgs());
   } else {
      // Create timer to let run CEF message loop together with ROOT event loop
      Int_t interval = gEnv->GetValue("WebGui.CefTimer", 10);
      TCefTimer *timer = new TCefTimer((interval > 0) ? interval : 10, kTRUE);
      timer->TurnOn();
   }

   // window not yet exists here
   return handle;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
/// Destructor
/// Closes browser window if any

RCefWebDisplayHandle::~RCefWebDisplayHandle()
{
   fValid = kInvalid;

   CloseBrowser();
}

///////////////////////////////////////////////////////////////////////////////////////////////////
/// Closes associated browser window

void RCefWebDisplayHandle::CloseBrowser()
{
   if (fBrowser && fCloseBrowser) {
      auto host = fBrowser->GetHost();
      if (host) host->CloseBrowser(true);
      fBrowser = nullptr;
   }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
/// Process system events until browser content is available
/// Used in headless mode for batch production like chrome --dump-dom is doing

bool RCefWebDisplayHandle::WaitForContent(int tmout_sec, const std::string &extra_args)
{
   int expired = tmout_sec * 100;
   bool did_try = false, print_finished = false;
   std::string pdffile;
   if (!extra_args.empty() && (extra_args.find("--print-to-pdf=")==0))
      pdffile = extra_args.substr(15);

   while ((--expired > 0) && GetContent().empty() && !print_finished) {

      if (gSystem->ProcessEvents()) break; // interrupted, has to return

      CefDoMessageLoopWork();

      if (fBrowser && !did_try && fBrowser->HasDocument() && !fBrowser->IsLoading() && fBrowser->GetMainFrame()) {
         did_try = true;
         if (pdffile.empty()) {
            fBrowser->GetMainFrame()->GetSource(new FrameSourceVisitor(this));
         } else {
            CefPdfPrintSettings settings;
            fBrowser->GetHost()->PrintToPDF(pdffile, settings, new HeadlessPrintCallback(&print_finished));
         }
      }

      gSystem->Sleep(10); // only 10 ms sleep
   }

   CloseBrowser();

   // call it once here to complete browser window closing, timer is not installed in batch mode
   CefDoMessageLoopWork();

   return !GetContent().empty();
}

///////////////////////////////////////////////////////////////////////////////////////////////////
/// Resize browser window

bool RCefWebDisplayHandle::Resize(int width, int height)
{
   if (!fBrowser)
      return false;
   return GuiHandler::PlatformResize(fBrowser, width, height);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
/// Add CEF creator

void RCefWebDisplayHandle::AddCreator()
{
   auto &entry = FindCreator("cef");
   if (!entry)
      GetMap().emplace("cef", std::make_unique<CefCreator>());
}

///////////////////////////////////////////////////////////////////////////////////////////////////
/// Helper struct to add creator

struct RCefCreatorReg {
   RCefCreatorReg() { RCefWebDisplayHandle::AddCreator(); }
} newRCefCreatorReg;

