// Copyright (c) 2013 The Chromium Embedded Framework Authors. All rights
// reserved. Use of this source code is governed by a BSD-style license that
// can be found in the LICENSE file.

#include "simple_app.h"

#include <string>

#include "gui_handler.h"
#include "osr_handler.h"
#include "include/cef_browser.h"
#include "include/cef_scheme.h"
#include "include/views/cef_browser_view.h"
#include "include/views/cef_window.h"
#include "include/wrapper/cef_helpers.h"
#include "include/wrapper/cef_stream_resource_handler.h"

#include "THttpServer.h"
#include "THttpCallArg.h"
#include "TUrl.h"
#include "TTimer.h"
#include "TApplication.h"
#include "TROOT.h"

THttpServer *gHandlingServer = 0;

// TODO: memory cleanup of these arguments
class TCefHttpCallArg : public THttpCallArg, public CefResourceHandler {
protected:
   // QWebEngineUrlRequestJob *fRequest;
   int fDD;

   CefRefPtr<CefCallback> callback_;

   int offset_;

public:
   TCefHttpCallArg() : THttpCallArg(), CefResourceHandler(), fDD(0), callback_(), offset_(0) {}

   virtual ~TCefHttpCallArg()
   {
      if (fDD != 1) printf("FAAAAAAAAAAAAAIL %d\n", fDD);
      fDD = -1;
   }

   // this is callback from HTTP server
   virtual void HttpReplied()
   {
      fDD++;

      if (IsFile()) {
         // send file
         Int_t content_len;
         char *file_content = THttpServer::ReadFileContent((const char *)GetContent(), content_len);
         SetBinData(file_content, content_len);
      }

      offset_ = 0;           // used for reading
      callback_->Continue(); // we have result and can continue with processing
   }

   void Cancel() OVERRIDE { CEF_REQUIRE_IO_THREAD(); }

   bool ProcessRequest(CefRefPtr<CefRequest> request, CefRefPtr<CefCallback> callback) OVERRIDE
   {
      CEF_REQUIRE_IO_THREAD();

      callback_ = callback;

      gHandlingServer->SubmitHttp(this);

      return true;
   }

   void GetResponseHeaders(CefRefPtr<CefResponse> response, int64 &response_length, CefString &redirectUrl) OVERRIDE
   {
      CEF_REQUIRE_IO_THREAD();

      DCHECK(!Is404());

      response->SetMimeType(GetContentType());
      response->SetStatus(200);

      // Set the resulting response length.
      response_length = GetContentLength();
   }

   bool ReadResponse(void *data_out, int bytes_to_read, int &bytes_read, CefRefPtr<CefCallback> callback) OVERRIDE
   {
      CEF_REQUIRE_IO_THREAD();

      bool has_data = false;
      bytes_read = 0;

      if (offset_ < GetContentLength()) {
         char *data_ = (char *)GetContent();
         // Copy the next block of data into the buffer.
         int transfer_size = GetContentLength() - offset_;
         if (transfer_size > bytes_to_read) transfer_size = bytes_to_read;
         memcpy(data_out, data_ + offset_, transfer_size);
         offset_ += transfer_size;

         bytes_read = transfer_size;
         has_data = true;
      }

      return has_data;
   }

   IMPLEMENT_REFCOUNTING(TCefHttpCallArg);
   DISALLOW_COPY_AND_ASSIGN(TCefHttpCallArg);
};

namespace {

// When using the Views framework this object provides the delegate
// implementation for the CefWindow that hosts the Views-based browser.
class SimpleWindowDelegate : public CefWindowDelegate {
public:
   explicit SimpleWindowDelegate(CefRefPtr<CefBrowserView> browser_view) : browser_view_(browser_view) {}

   void OnWindowCreated(CefRefPtr<CefWindow> window) OVERRIDE
   {
      // Add the browser view and show the window.
      window->AddChildView(browser_view_);
      window->Show();

      // Give keyboard focus to the browser view.
      browser_view_->RequestFocus();
   }

   void OnWindowDestroyed(CefRefPtr<CefWindow> window) OVERRIDE { browser_view_ = NULL; }

   bool CanClose(CefRefPtr<CefWindow> window) OVERRIDE
   {
      // Allow the window to close if the browser says it's OK.
      CefRefPtr<CefBrowser> browser = browser_view_->GetBrowser();
      if (browser) return browser->GetHost()->TryCloseBrowser();
      return true;
   }

private:
   CefRefPtr<CefBrowserView> browser_view_;

   IMPLEMENT_REFCOUNTING(SimpleWindowDelegate);
   DISALLOW_COPY_AND_ASSIGN(SimpleWindowDelegate);
};

class ROOTSchemeHandlerFactory : public CefSchemeHandlerFactory {
protected:
public:
   explicit ROOTSchemeHandlerFactory() : CefSchemeHandlerFactory() {}

   virtual CefRefPtr<CefResourceHandler> Create(CefRefPtr<CefBrowser> browser, CefRefPtr<CefFrame> frame,
                                                const CefString &scheme_name, CefRefPtr<CefRequest> request)
   {
      std::string addr = request->GetURL().ToString();

      // printf("Request %s server %p\n", addr.c_str(), gHandlingServer);

      TUrl url(addr.c_str());

      const char *inp_path = url.GetFile();
      const char *inp_query = url.GetOptions();
      const char *inp_method = "GET";

      TString fname;

      if (gHandlingServer->IsFileRequested(inp_path, fname)) {
         // process file - no need for special requests handling

         printf("Send file %s\n", fname.Data());

         const char *mime = THttpServer::GetMimeType(fname.Data());

         Int_t content_len;

         char *content = THttpServer::ReadFileContent(fname.Data(), content_len);

         std::string str_content;
         str_content.append(content, content_len);

         free(content);

         // Create a stream reader for |html_content|.
         CefRefPtr<CefStreamReader> stream = CefStreamReader::CreateForData(
            static_cast<void *>(const_cast<char *>(str_content.c_str())), str_content.size());

         // Constructor for HTTP status code 200 and no custom response headers.
         // There’s also a version of the constructor for custom status code and response headers.
         return new CefStreamResourceHandler(mime, stream);
      }

      TCefHttpCallArg *arg = new TCefHttpCallArg();
      arg->SetPathAndFileName(inp_path);
      arg->SetQuery(inp_query);
      arg->SetMethod(inp_method);
      arg->SetTopName("webgui");

      // gHandlingServer->SubmitHttp(arg);

      // just return handler
      return arg;
   }

   IMPLEMENT_REFCOUNTING(ROOTSchemeHandlerFactory);
};

} // namespace

SimpleApp::SimpleApp(const std::string &url, const std::string &cef_main, THttpServer *serv, bool isbatch)
   : CefApp(), CefBrowserProcessHandler(), /*CefRenderProcessHandler(),*/ fUrl(), fCefMain(cef_main), fBatch(isbatch)
{
   fUrl = "rootscheme://rootserver";
   fUrl.append(url);
   gHandlingServer = serv;
   printf("Assign url %s\n", fUrl.c_str());
}

SimpleApp::~SimpleApp()
{
}

void SimpleApp::OnRegisterCustomSchemes(CefRawPtr<CefSchemeRegistrar> registrar)
{
   // registrar->AddCustomScheme("rootscheme", true, true, true, true, true, true);
   registrar->AddCustomScheme("rootscheme", true, false, false, true, false, false);
}

void SimpleApp::OnBeforeCommandLineProcessing(const CefString &process_type, CefRefPtr<CefCommandLine> command_line)
{
   std::string name = process_type.ToString();
   std::string prog = command_line->GetProgram().ToString();

   printf("Start process %s %s\n", name.c_str(), prog.c_str());
}

void SimpleApp::OnBeforeChildProcessLaunch(CefRefPtr<CefCommandLine> command_line)
{

   std::string newprog = fCefMain;
   command_line->SetProgram(newprog);
   std::string prog = command_line->GetProgram().ToString();

   // if (fBatch) command_line->AppendArgument("--root-batch");
   // printf("OnBeforeChildProcessLaunch %s\n", prog.c_str());
}

/*
void SimpleApp::OnContextCreated(CefRefPtr<CefBrowser> browser, CefRefPtr<CefFrame> frame,
                                 CefRefPtr<CefV8Context> context)
{
   printf("$$$$$$$$$ SimpleApp::OnContextCreated\n");

   if (!fBatch) return;

   // Retrieve the context's window object.
   CefRefPtr<CefV8Value> object = context->GetGlobal();

   // Create a new V8 string value. See the "Basic JS Types" section below.
   CefRefPtr<CefV8Value> str = CefV8Value::CreateString("My Value!");

   // Add the string to the window object as "window.myval". See the "JS Objects" section below.
   object->SetValue("ROOT_BATCH_FLAG", str, V8_PROPERTY_ATTRIBUTE_NONE);

   printf("ADD BATCH FALG\n");
}

void SimpleApp::OnBrowserCreated(CefRefPtr<CefBrowser> browser)
{
   printf("$$$$$$$$$ SimpleApp::OnBrowserCreated\n");
}
*/

void SimpleApp::OnContextInitialized()
{
   CEF_REQUIRE_UI_THREAD();

   CefRegisterSchemeHandlerFactory("rootscheme", "rootserver", new ROOTSchemeHandlerFactory());

   // Specify CEF browser settings here.
   CefBrowserSettings browser_settings;

   if (fBatch) {

      CefRefPtr<OsrHandler> handler(new OsrHandler(gHandlingServer));

      CefWindowInfo window_info;

      window_info.SetAsWindowless(0);

      printf("Create OSR browser %s\n", fUrl.c_str());

      // Create the first browser window.
      CefBrowserHost::CreateBrowser(window_info, handler, fUrl, browser_settings, NULL);

      return;
   }

#if defined(OS_WIN) || defined(OS_LINUX)
   // Create the browser using the Views framework if "--use-views" is specified
   // via the command-line. Otherwise, create the browser using the native
   // platform framework. The Views framework is currently only supported on
   // Windows and Linux.

   CefRefPtr<CefCommandLine> command_line = CefCommandLine::GetGlobalCommandLine();

   bool use_views = command_line->HasSwitch("use-views");
#else
   bool use_views = false;
#endif

   // SimpleHandler implements browser-level callbacks.
   CefRefPtr<GuiHandler> handler(new GuiHandler(gHandlingServer, use_views));

   if (use_views) {
      // Create the BrowserView.
      CefRefPtr<CefBrowserView> browser_view =
         CefBrowserView::CreateBrowserView(handler, fUrl, browser_settings, NULL, NULL);

      // Create the Window. It will show itself after creation.
      CefWindow::CreateTopLevelWindow(new SimpleWindowDelegate(browser_view));
   } else {
      // Information used when creating the native window.
      CefWindowInfo window_info;

#if defined(OS_WIN)
      // On Windows we need to specify certain flags that will be passed to
      // CreateWindowEx().
      window_info.SetAsPopup(NULL, "cefsimple");
#endif

      // Create the first browser window.
      CefBrowserHost::CreateBrowser(window_info, handler, fUrl, browser_settings, NULL);
   }
}

class TCefTimer : public TTimer {
public:
   TCefTimer(Long_t milliSec, Bool_t mode) : TTimer(milliSec, mode)
   {
      // construtor
   }
   virtual ~TCefTimer()
   {
      // destructor
   }
   virtual void Timeout()
   {
      // just dummy workaround
      CefDoMessageLoopWork();
   }
};

extern "C" void webgui_start_browser_in_cef3(const char *url, void *http_serv, bool batch_mode, const char *rootsys,
                                             const char *cef_path)
{
   TApplication *root_app = gROOT->GetApplication();

   CefMainArgs main_args(root_app->Argc(), root_app->Argv());

   // CEF applications have multiple sub-processes (render, plugin, GPU, etc)
   // that share the same executable. This function checks the command-line and,
   // if this is a sub-process, executes the appropriate logic.

   /*         int exit_code = CefExecuteProcess(main_args, NULL, NULL);
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

   TString path, path2, cef_main;
   path.Form("%s/Resources/", cef_path);
   path2.Form("%s/Resources/locales/", cef_path);
   cef_main.Form("%s/bin/cef_main", rootsys);

   cef_string_ascii_to_utf16(path.Data(), path.Length(), &settings.resources_dir_path);

   cef_string_ascii_to_utf16(path2.Data(), path2.Length(), &settings.locales_dir_path);

   settings.no_sandbox = true;
   // settings.single_process = true;

   if (batch_mode) settings.windowless_rendering_enabled = true;

   //TString plog = "cef.log";
   //cef_string_ascii_to_utf16(plog.Data(), plog.Length(), &settings.log_file);
   settings.log_severity = LOGSEVERITY_INFO; // LOGSEVERITY_VERBOSE;
   //settings.uncaught_exception_stack_size = 100;
   //settings.ignore_certificate_errors = true;

   // SimpleApp implements application-level callbacks for the browser process.
   // It will create the first browser instance in OnContextInitialized() after
   // CEF has initialized.
   CefRefPtr<SimpleApp> *app =
      new CefRefPtr<SimpleApp>(new SimpleApp(url, cef_main.Data(), (THttpServer *)http_serv, batch_mode));

   // Initialize CEF for the browser process.
   CefInitialize(main_args, settings, app->get(), NULL);

   // let run CEF message loop, should be improved later
   TCefTimer *timer = new TCefTimer(10, kTRUE);
   timer->TurnOn();

   // Shut down CEF.
   // CefShutdown();
}
