/// \file RCefWebDisplayHandle.cxx
/// \ingroup WebGui ROOT7
/// \author Sergey Linev <s.linev@gsi.de>
/// \date 2020-08-21
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
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
#include "TEnv.h"
#include "TApplication.h"

#include <ROOT/RMakeUnique.hxx>
#include <ROOT/RLogger.hxx>


class TCefTimer : public TTimer {
public:
   TCefTimer(Long_t milliSec, Bool_t mode) : TTimer(milliSec, mode) {}
   void Timeout() override
   {
      // just let run loop
      CefDoMessageLoopWork();

   }
};


std::unique_ptr<ROOT::Experimental::RWebDisplayHandle> RCefWebDisplayHandle::CefCreator::Display(const ROOT::Experimental::RWebDisplayArgs &args)
{

   auto handle = std::make_unique<RCefWebDisplayHandle>(args.GetFullUrl());

   if (fCefApp) {
      if (SimpleApp::GetHttpServer() != args.GetHttpServer()) {
         R__ERROR_HERE("CEF") << "CEF do not allows to use different THttpServer instances";
         return nullptr;
      }

      fCefApp->SetNextHandle(handle.get());

      CefRect rect((args.GetX() > 0) ? args.GetX() : 0, (args.GetY() > 0) ? args.GetY() : 0,
            (args.GetWidth() > 0) ? args.GetWidth() : 800, (args.GetHeight() > 0) ? args.GetHeight() : 600);
      fCefApp->StartWindow(args.GetFullUrl(), args.IsHeadless(), rect);
      return handle;
   }

   TApplication *root_app = gROOT->GetApplication();

#if defined(OS_WIN)
   CefMainArgs main_args(GetModuleHandle(nullptr));
#else
   CefMainArgs main_args(root_app->Argc(), root_app->Argv());
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

   // Specify CEF global settings here.
   CefSettings settings;

   TString cef_main = TROOT::GetBinDir() + "/cef_main";

   // cef_string_ascii_to_utf16(path.Data(), path.Length(), &settings.resources_dir_path);
   // cef_string_ascii_to_utf16(path2.Data(), path2.Length(), &settings.locales_dir_path);

   settings.no_sandbox = true;
   // if (gROOT->IsWebDisplayBatch()) settings.single_process = true;

   // if (batch_mode)
   // settings.windowless_rendering_enabled = true;

   // settings.external_message_pump = true;
   // settings.multi_threaded_message_loop = false;

   std::string plog = "cef.log";
   cef_string_ascii_to_utf16(plog.c_str(), plog.length(), &settings.log_file);

   settings.log_severity = LOGSEVERITY_ERROR; // LOGSEVERITY_VERBOSE, LOGSEVERITY_INFO, LOGSEVERITY_WARNING,
   // LOGSEVERITY_ERROR, LOGSEVERITY_DISABLE
   // settings.uncaught_exception_stack_size = 100;
   // settings.ignore_certificate_errors = true;

   // settings.remote_debugging_port = 7890;

   SimpleApp::SetHttpServer(args.GetHttpServer());

   // SimpleApp implements application-level callbacks for the browser process.
   // It will create the first browser instance in OnContextInitialized() after
   // CEF has initialized.
   fCefApp = new SimpleApp(cef_main.Data(), args.GetFullUrl(), args.IsHeadless(), args.GetWidth(), args.GetHeight());

   fCefApp->SetNextHandle(handle.get());

   // Initialize CEF for the browser process.
   CefInitialize(main_args, settings, fCefApp.get(), nullptr);

   Int_t interval = gEnv->GetValue("WebGui.CefTimer", 10);
   // let run CEF message loop, should be improved later
   TCefTimer *timer = new TCefTimer((interval > 0) ? interval : 10, kTRUE);
   timer->TurnOn();

   // window not yet exists here
   return handle;
}


RCefWebDisplayHandle::~RCefWebDisplayHandle()
{
   fValid = kInvalid;

   if (fBrowser) {
      auto host = fBrowser->GetHost();
      if (host) host->CloseBrowser(true);
      fBrowser = nullptr;
   }

}

void RCefWebDisplayHandle::AddCreator()
{
   auto &entry = FindCreator("cef");
   if (!entry)
      GetMap().emplace("cef", std::make_unique<CefCreator>());
}


struct RCefCreatorReg {
   RCefCreatorReg() { RCefWebDisplayHandle::AddCreator(); }
} newRCefCreatorReg;

