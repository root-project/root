// Copyright (c) 2013 The Chromium Embedded Framework Authors. All rights
// reserved. Use of this source code is governed by a BSD-style license that
// can be found in the LICENSE file.

#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wshadow"

#include "simple_app.h"

#include <string>
#include <cstdio>

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
#include "TBase64.h"

THttpServer *gHandlingServer = nullptr;

CefRefPtr<SimpleApp> *gCefApp = nullptr;

class TCefHttpCallArg : public THttpCallArg {
protected:

   CefRefPtr<CefCallback> fCallBack{nullptr};

public:
   explicit TCefHttpCallArg() = default;

   void AssignCallback(CefRefPtr<CefCallback> cb) { fCallBack = cb; }

   // this is callback from HTTP server
   virtual void HttpReplied()
   {
      if (IsFile()) {
         // send file
         std::string file_content = THttpServer::ReadFileContent((const char *)GetContent());
         SetContent(std::move(file_content));
      }
//      else {
//         printf("CEF Request replied %s %s %s  len: %d %s\n", GetPathName(), GetFileName(), GetQuery(),
//               GetContentLength(), (const char *) GetContent() );
//      }

      fCallBack->Continue(); // we have result and can continue with processing
   }
};

class TGuiResourceHandler : public CefResourceHandler {
public:
   // QWebEngineUrlRequestJob *fRequest;
   std::shared_ptr<TCefHttpCallArg> fArg;

   int fTransferOffset{0};

   explicit TGuiResourceHandler()
   {
      fArg = std::make_shared<TCefHttpCallArg>();
   }

   virtual ~TGuiResourceHandler() {}

   void Cancel() OVERRIDE { CEF_REQUIRE_IO_THREAD(); }

   bool ProcessRequest(CefRefPtr<CefRequest> request, CefRefPtr<CefCallback> callback) OVERRIDE
   {
      CEF_REQUIRE_IO_THREAD();

      fArg->AssignCallback(callback);

      gHandlingServer->SubmitHttp(fArg);

      return true;
   }

   void GetResponseHeaders(CefRefPtr<CefResponse> response, int64 &response_length, CefString &redirectUrl) OVERRIDE
   {
      CEF_REQUIRE_IO_THREAD();

      if (fArg->Is404()) {
         response->SetMimeType("text/html");
         response->SetStatus(404);
         response_length = 0;
      } else {
         response->SetMimeType(fArg->GetContentType());
         response->SetStatus(200);
         response_length = fArg->GetContentLength();

         if (fArg->NumHeader() > 0) {
            //printf("******* Response with extra headers\n");
            CefResponse::HeaderMap headers;
            for (Int_t n = 0; n < fArg->NumHeader(); ++n) {
               TString name = fArg->GetHeaderName(n);
               TString value = fArg->GetHeader(name.Data());
               headers.emplace(CefString(name.Data()), CefString(value.Data()));
               //printf("   header %s %s\n", name.Data(), value.Data());
            }
            response->SetHeaderMap(headers);
         }
//         if (strstr(fArg->GetQuery(),"connection="))
//            printf("Reply %s %s %s  len: %d %s\n", fArg->GetPathName(), fArg->GetFileName(), fArg->GetQuery(),
//                  fArg->GetContentLength(), (const char *) fArg->GetContent() );
      }
      // DCHECK(!fArg->Is404());
   }

   bool ReadResponse(void *data_out, int bytes_to_read, int &bytes_read, CefRefPtr<CefCallback> callback) OVERRIDE
   {
      CEF_REQUIRE_IO_THREAD();

      if (!fArg) return false;

      bytes_read = 0;

      if (fTransferOffset < fArg->GetContentLength()) {
         char *data_ = (char *)fArg->GetContent();
         // Copy the next block of data into the buffer.
         int transfer_size = fArg->GetContentLength() - fTransferOffset;
         if (transfer_size > bytes_to_read)
            transfer_size = bytes_to_read;
         memcpy(data_out, data_ + fTransferOffset, transfer_size);
         fTransferOffset += transfer_size;

         bytes_read = transfer_size;
      }

      // if content fully copied - can release reference, object will be cleaned up
      if (fTransferOffset >= fArg->GetContentLength())
         fArg.reset();

      return bytes_read > 0;
   }

   IMPLEMENT_REFCOUNTING(TGuiResourceHandler);
   DISALLOW_COPY_AND_ASSIGN(TGuiResourceHandler);
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

   void OnWindowDestroyed(CefRefPtr<CefWindow> window) OVERRIDE { browser_view_ = nullptr; }

   bool CanClose(CefRefPtr<CefWindow> window) OVERRIDE
   {
      // Allow the window to close if the browser says it's OK.
      CefRefPtr<CefBrowser> browser = browser_view_->GetBrowser();
      if (browser)
         return browser->GetHost()->TryCloseBrowser();
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

      TString inp_query = url.GetOptions();

      TString fname;

      if (gHandlingServer->IsFileRequested(inp_path, fname)) {
         // process file - no need for special requests handling

         // printf("Sending file %s via cef\n", fname.Data());

         const char *mime = THttpServer::GetMimeType(fname.Data());

         std::string str_content = THttpServer::ReadFileContent(fname.Data());

         // Create a stream reader for |html_content|.
         CefRefPtr<CefStreamReader> stream = CefStreamReader::CreateForData(
            static_cast<void *>(const_cast<char *>(str_content.c_str())), str_content.size());

         // Constructor for HTTP status code 200 and no custom response headers.
         // There’s also a version of the constructor for custom status code and response headers.
         return new CefStreamResourceHandler(mime, stream);
      }

      std::string inp_method = request->GetMethod().ToString();

      TGuiResourceHandler *handler = new TGuiResourceHandler();
      handler->fArg->SetMethod(inp_method.c_str());
      handler->fArg->SetPathAndFileName(inp_path);
      handler->fArg->SetTopName("webgui");

      // printf("Method %s Request %s %s\n", inp_method.c_str(), inp_path, inp_query.Data());

      if (inp_method == "POST") {

         CefRefPtr< CefPostData > post_data = request->GetPostData();

         if (!post_data) {
            printf("FATAL - NO POST DATA in CEF HANDLER!!!\n");
            exit(1);
         } else {
            CefPostData::ElementVector elements;
            post_data->GetElements(elements);
            size_t sz = 0, off = 0;
            for (unsigned n = 0; n < elements.size(); ++n)
               sz += elements[n]->GetBytesCount();
            std::string data;
            data.resize(sz);

            for (unsigned n = 0; n < elements.size(); ++n) {
               sz = elements[n]->GetBytes(elements[n]->GetBytesCount(), (char *)data.data() + off);
               off += sz;
            }
            // printf("Get post data %u  %u %s\n", (unsigned)off, (unsigned)data.length(), data.c_str());
            handler->fArg->SetPostData(std::move(data));
         }
      } else if (inp_query.Index("&post=") != kNPOS) {
         Int_t pos = inp_query.Index("&post=");
         TString buf = TBase64::Decode(inp_query.Data() + pos + 6);
         handler->fArg->SetPostData(std::string(buf.Data()));
         inp_query.Resize(pos);
      }

      handler->fArg->SetQuery(inp_query.Data());

      // just return handler
      return handler;
   }

   IMPLEMENT_REFCOUNTING(ROOTSchemeHandlerFactory);
};

} // namespace

SimpleApp::SimpleApp(const std::string &url, const std::string &cef_main, THttpServer *serv, bool isbatch)
   : CefApp(), CefBrowserProcessHandler(), /*CefRenderProcessHandler(),*/ fUrl(url), fCefMain(cef_main),
     fBatch(isbatch), fRect(), fOsrHandler(), fUseViewes(false), fGuiHandler()
{
   gHandlingServer = serv;
}

SimpleApp::~SimpleApp()
{
}

void SimpleApp::OnRegisterCustomSchemes(CefRawPtr<CefSchemeRegistrar> registrar)
{
   // registrar->AddCustomScheme("rootscheme", true, true, true, true, true, true);
  // registrar->AddCustomScheme("rootscheme", true, false, false, true, false, false);
}

void SimpleApp::OnBeforeCommandLineProcessing(const CefString &process_type, CefRefPtr<CefCommandLine> command_line)
{
//   std::string name = process_type.ToString();
//   std::string prog = command_line->GetProgram().ToString();
//   printf("OnBeforeCommandLineProcessing %s %s\n", name.c_str(), prog.c_str());
//   if (fBatch) {
//      command_line->AppendSwitch("disable-gpu");
//      command_line->AppendSwitch("disable-gpu-compositing");
//      command_line->AppendSwitch("disable-gpu-sandbox");
//   }
}

void SimpleApp::OnBeforeChildProcessLaunch(CefRefPtr<CefCommandLine> command_line)
{
   std::string newprog = fCefMain;
   command_line->SetProgram(newprog);

   // printf("OnBeforeChildProcessLaunch %s\n", command_line->GetProgram().ToString().c_str());
   if (fBatch) {
      command_line->AppendSwitch("disable-webgl");
      command_line->AppendSwitch("disable-gpu");
      command_line->AppendSwitch("disable-gpu-compositing");
//      command_line->AppendSwitch("disable-gpu-sandbox");
   }
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

   CefRegisterSchemeHandlerFactory("http", "rootserver.local", new ROOTSchemeHandlerFactory());

   StartWindow(fUrl, fBatch, fRect);
}

void SimpleApp::StartWindow(const std::string &addr, bool batch, CefRect &rect)
{
   CEF_REQUIRE_UI_THREAD();

   std::string url = "http://rootserver.local";
   url.append(addr);
   if (url.find("?") != std::string::npos)
      url.append("&");
   else
      url.append("?");

   url.append("platform=cef3&ws=longpoll");

   // Specify CEF browser settings here.
   CefBrowserSettings browser_settings;

   CefWindowInfo window_info;

#if defined(OS_WIN)
   RECT wnd_rect = {rect.x, rect.y, rect.x + rect.width, rect.y + rect.height};
   if (!rect.IsEmpty()) window_info.SetAsChild(0, wnd_rect);
#elif defined(OS_LINUX)
   if (!rect.IsEmpty()) window_info.SetAsChild(0, rect);
#else
   if (!rect.IsEmpty()) window_info.SetAsChild(0, rect.x, rect.y, rect.width, rect.height );
#endif

   if (batch) {

      if (!fOsrHandler)
         fOsrHandler = new OsrHandler(gHandlingServer);
      // CefRefPtr<OsrHandler> handler(new OsrHandler(gHandlingServer));

      window_info.SetAsWindowless(0);

      printf("Create OSR browser %s\n", url.c_str());

      // Create the first browser window.
      CefBrowserHost::CreateBrowser(window_info, fOsrHandler, url, browser_settings, nullptr);

      return;
   }

#if defined(OS_WIN) || defined(OS_LINUX)
   // Create the browser using the Views framework if "--use-views" is specified
   // via the command-line. Otherwise, create the browser using the native
   // platform framework. The Views framework is currently only supported on
   // Windows and Linux.

   CefRefPtr<CefCommandLine> command_line = CefCommandLine::GetGlobalCommandLine();

// bool use_views = command_line->HasSwitch("use-views");
#else
// bool use_views = false;
#endif

   if (!fGuiHandler) {
      fUseViewes = false;
      fGuiHandler = new GuiHandler(gHandlingServer, fUseViewes);
   }

   // SimpleHandler implements browser-level callbacks.
   // CefRefPtr<GuiHandler> handler(new GuiHandler(gHandlingServer, use_views));

   if (fUseViewes) {
      // Create the BrowserView.
      CefRefPtr<CefBrowserView> browser_view =
         CefBrowserView::CreateBrowserView(fGuiHandler, url, browser_settings, nullptr, nullptr);

      // Create the Window. It will show itself after creation.
      CefWindow::CreateTopLevelWindow(new SimpleWindowDelegate(browser_view));
   } else {

#if defined(OS_WIN)
      // On Windows we need to specify certain flags that will be passed to
      // CreateWindowEx().
      window_info.SetAsPopup(0, "cefsimple");
#endif

      // Create the first browser window.
      CefBrowserHost::CreateBrowser(window_info, fGuiHandler, url, browser_settings, nullptr);
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
                                             const char *cef_path, unsigned width, unsigned height)
{
   if (gCefApp) {
      // printf("Starting next CEF window\n");

      if (gHandlingServer != (THttpServer *)http_serv) {
         printf("CEF plugin do not allow to use other THttpServer instance\n");
      } else {
         CefRect rect(0, 0, width, height);
         gCefApp->get()->StartWindow(url, batch_mode, rect);
      }
      return;
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

   TString path, path2, cef_main;
   path.Form("%s/Resources/", cef_path);
   path2.Form("%s/Resources/locales/", cef_path);
   cef_main.Form("%s/bin/cef_main", rootsys);

   cef_string_ascii_to_utf16(path.Data(), path.Length(), &settings.resources_dir_path);

   cef_string_ascii_to_utf16(path2.Data(), path2.Length(), &settings.locales_dir_path);

   settings.no_sandbox = true;
   if (gROOT->IsWebDisplayBatch()) settings.single_process = true;

   //if (batch_mode)
   settings.windowless_rendering_enabled = true;

   TString plog = "cef.log";
   cef_string_ascii_to_utf16(plog.Data(), plog.Length(), &settings.log_file);
   settings.log_severity = LOGSEVERITY_INFO; // LOGSEVERITY_VERBOSE;
   // settings.uncaught_exception_stack_size = 100;
   // settings.ignore_certificate_errors = true;
   // settings.remote_debugging_port = 7890;

   // SimpleApp implements application-level callbacks for the browser process.
   // It will create the first browser instance in OnContextInitialized() after
   // CEF has initialized.
   gCefApp = new CefRefPtr<SimpleApp>(new SimpleApp(url, cef_main.Data(), (THttpServer *)http_serv, batch_mode));
   gCefApp->get()->SetRect(width, height);

   // Initialize CEF for the browser process.
   CefInitialize(main_args, settings, gCefApp->get(), nullptr);

   // let run CEF message loop, should be improved later
   TCefTimer *timer = new TCefTimer(10, kTRUE);
   timer->TurnOn();

   // Shut down CEF.
   // CefShutdown();
}
