/// \file simple_app.cxx
/// \ingroup WebGui
/// \author Sergey Linev <S.Linev@gsi.de>
/// \date 2017-06-29
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

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


#if !defined(_MSC_VER)
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wshadow"
#endif

#include "simple_app.h"

#include <string>
#include <cstdio>

#include "include/cef_browser.h"
#include "include/cef_scheme.h"
#include "include/cef_version.h"
#include "include/views/cef_browser_view.h"
#include "include/views/cef_window.h"
#include "include/wrapper/cef_helpers.h"
#include "include/wrapper/cef_stream_resource_handler.h"

#include "THttpServer.h"
#include "THttpCallArg.h"
#include "TUrl.h"
#include "TTimer.h"
#include "TSystem.h"
#include "TBase64.h"

#include <memory>

#include "RCefWebDisplayHandle.hxx"
#include <ROOT/RLogger.hxx>

THttpServer *SimpleApp::gHttpServer = nullptr;

THttpServer *SimpleApp::GetHttpServer() { return gHttpServer; }
void SimpleApp::SetHttpServer(THttpServer *serv) { gHttpServer = serv; }


class TCefHttpCallArg : public THttpCallArg {
protected:

   CefRefPtr<CefCallback> fCallBack{nullptr};

   void CheckWSPageContent(THttpWSHandler *) override
   {
      std::string search = "JSROOT.ConnectWebWindow({";
      std::string replace = search + "platform:\"cef3\",socket_kind:\"longpoll\",";

      ReplaceAllinContent(search, replace, true);
   }

public:
   explicit TCefHttpCallArg() = default;

   void AssignCallback(CefRefPtr<CefCallback> cb) { fCallBack = cb; }

   // this is callback from HTTP server
   void HttpReplied() override
   {
      if (IsFile()) {
         // send file
         std::string file_content = THttpServer::ReadFileContent((const char *)GetContent());
         SetContent(std::move(file_content));
      }

      fCallBack->Continue(); // we have result and can continue with processing
   }
};

class TGuiResourceHandler : public CefResourceHandler {
public:
   // QWebEngineUrlRequestJob *fRequest;
   std::shared_ptr<TCefHttpCallArg> fArg;

   int fTransferOffset{0};

   explicit TGuiResourceHandler(bool dummy = false)
   {
      if (!dummy)
         fArg = std::make_shared<TCefHttpCallArg>();
   }

   virtual ~TGuiResourceHandler() {}

   void Cancel() OVERRIDE { CEF_REQUIRE_IO_THREAD(); }

   bool ProcessRequest(CefRefPtr<CefRequest> request, CefRefPtr<CefCallback> callback) OVERRIDE
   {
      CEF_REQUIRE_IO_THREAD();

      if (fArg) {
         fArg->AssignCallback(callback);
         SimpleApp::GetHttpServer()->SubmitHttp(fArg);
      } else {
         callback->Continue();
      }

      return true;
   }

   void GetResponseHeaders(CefRefPtr<CefResponse> response, int64 &response_length, CefString &redirectUrl) OVERRIDE
   {
      CEF_REQUIRE_IO_THREAD();

      if (!fArg || fArg->Is404()) {
         response->SetMimeType("text/html");
         response->SetStatus(404);
         response_length = 0;
      } else {
         response->SetMimeType(fArg->GetContentType());
         response->SetStatus(200);
         response_length = fArg->GetContentLength();

         if (fArg->NumHeader() > 0) {
            // printf("******* Response with extra headers\n");
            CefResponse::HeaderMap headers;
            for (Int_t n = 0; n < fArg->NumHeader(); ++n) {
               TString name = fArg->GetHeaderName(n);
               TString value = fArg->GetHeader(name.Data());
               headers.emplace(CefString(name.Data()), CefString(value.Data()));
               // printf("   header %s %s\n", name.Data(), value.Data());
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
   explicit ROOTSchemeHandlerFactory() = default;

   CefRefPtr<CefResourceHandler> Create(CefRefPtr<CefBrowser> browser, CefRefPtr<CefFrame> frame,
                                        const CefString &scheme_name, CefRefPtr<CefRequest> request) OVERRIDE
   {
      std::string addr = request->GetURL().ToString();

      TUrl url(addr.c_str());

      const char *inp_path = url.GetFile();

      TString inp_query = url.GetOptions();

      TString fname;

      if (SimpleApp::GetHttpServer()->IsFileRequested(inp_path, fname)) {
         // process file - no need for special requests handling

         // when file not exists - return nullptr
         if (gSystem->AccessPathName(fname.Data()))
            return new TGuiResourceHandler(true);

         const char *mime = THttpServer::GetMimeType(fname.Data());

         CefRefPtr<CefStreamReader> stream = CefStreamReader::CreateForFile(fname.Data());

         // Constructor for HTTP status code 200 and no custom response headers.
         // There’s also a version of the constructor for custom status code and response headers.
         return new CefStreamResourceHandler(mime, stream);
      }

      std::string inp_method = request->GetMethod().ToString();

      // printf("REQUEST METHOD %s\n", inp_method.c_str());

      TGuiResourceHandler *handler = new TGuiResourceHandler();
      handler->fArg->SetMethod(inp_method.c_str());
      handler->fArg->SetPathAndFileName(inp_path);
      handler->fArg->SetTopName("webgui");

      if (inp_method == "POST") {

         CefRefPtr< CefPostData > post_data = request->GetPostData();

         if (!post_data) {
            R__ERROR_HERE("CEF") << "FATAL - NO POST DATA in CEF HANDLER!!!";
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

SimpleApp::SimpleApp(const std::string &cef_main, const std::string &url, bool isbatch, int width, int height)
   : CefApp(), CefBrowserProcessHandler(), /*CefRenderProcessHandler(),*/ fCefMain(cef_main), fFirstUrl(url), fFirstBatch(isbatch)
{
   fFirstRect.Set(0, 0, width, height);
}

void SimpleApp::OnRegisterCustomSchemes(CefRawPtr<CefSchemeRegistrar> registrar)
{
   // registrar->AddCustomScheme("rootscheme", true, true, true, true, true, true);
  // registrar->AddCustomScheme("rootscheme", true, false, false, true, false, false);
}

void SimpleApp::OnBeforeCommandLineProcessing(const CefString &process_type, CefRefPtr<CefCommandLine> command_line)
{
   std::string name = process_type.ToString();
   std::string prog = command_line->GetProgram().ToString();
   // printf("OnBeforeCommandLineProcessing %s %s\n", name.c_str(), prog.c_str());
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

   // printf("OnBeforeChildProcessLaunch %s LastBatch %s\n", command_line->GetProgram().ToString().c_str(), fLastBatch ? "true" : "false");

   if (fLastBatch) {
      command_line->AppendSwitch("disable-webgl");
      command_line->AppendSwitch("disable-gpu");
      command_line->AppendSwitch("disable-gpu-compositing");
//      command_line->AppendSwitch("disable-gpu-sandbox");
   }

   // auto str = command_line->GetCommandLineString().ToString();
   // printf("RUN %s\n", str.c_str());
}

void SimpleApp::OnContextInitialized()
{
   CEF_REQUIRE_UI_THREAD();

   CefRegisterSchemeHandlerFactory("http", "rootserver.local", new ROOTSchemeHandlerFactory());

   if (!fFirstUrl.empty())
      StartWindow(fFirstUrl, fFirstBatch, fFirstRect);

   printf("StartWindow completed!!\n");

}



class MyVisitor : public CefStringVisitor {
public:

   MyVisitor() : CefStringVisitor() {}

   virtual ~MyVisitor() {}

   void Visit( const CefString& str ) override
   {
      printf("Visit content len = %ld\n", str.length());

      std::string s = str.ToString();

      printf("Visit content substr(1000) = %s\n", s.substr(0,1000).c_str());

   }
private:
   // Include the default reference counting implementation.
   IMPLEMENT_REFCOUNTING(MyVisitor);
   DISALLOW_COPY_AND_ASSIGN(MyVisitor);
};



void SimpleApp::StartWindow(const std::string &addr, bool batch, CefRect &rect)
{
   CEF_REQUIRE_UI_THREAD();

   fLastBatch = false; // batch;

   std::string url;

   // TODO: later one should be able both remote and local at the same time
   if (SimpleApp::GetHttpServer()) {
      url = "http://rootserver.local";
      url.append(addr);
   } else {
      url = addr;
   }

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

   if (false) { // disable OSR handler for the moment, maybe later

      if (!fOsrHandler)
         fOsrHandler = new OsrHandler(GetHttpServer());

      window_info.SetAsWindowless(0);

#if CEF_COMMIT_NUMBER > 1934
      // Create the first browser window.
      CefBrowserHost::CreateBrowser(window_info, fOsrHandler, url, browser_settings, nullptr, nullptr);
#else
      CefBrowserHost::CreateBrowser(window_info, fOsrHandler, url, browser_settings, nullptr);
#endif

      return;
   }

#if defined(OS_WIN) || defined(OS_LINUX)
   // Create the browser using the Views framework if "--use-views" is specified
   // via the command-line. Otherwise, create the browser using the native
   // platform framework. The Views framework is currently only supported on
   // Windows and Linux.
#else
   fUseViewes = false;
#endif

   if (!fGuiHandler)
      fGuiHandler = new GuiHandler(GetHttpServer(), fUseViewes);

   if (fUseViewes) {
      // Create the BrowserView.
      CefRefPtr<CefBrowserView> browser_view =
#if CEF_COMMIT_NUMBER > 1934
         CefBrowserView::CreateBrowserView(fGuiHandler, url, browser_settings, nullptr, nullptr, nullptr);
#else
         CefBrowserView::CreateBrowserView(fGuiHandler, url, browser_settings, nullptr, nullptr);
#endif
      // Create the Window. It will show itself after creation.
      CefWindow::CreateTopLevelWindow(new SimpleWindowDelegate(browser_view));
   } else {

#if defined(OS_WIN)
      // On Windows we need to specify certain flags that will be passed to
      // CreateWindowEx().
      window_info.SetAsPopup(0, "cefsimple");
#endif

      // Create the first browser window.
#if CEF_COMMIT_NUMBER > 1934
      if (batch) {
         auto browser = CefBrowserHost::CreateBrowserSync(window_info, fGuiHandler, url, browser_settings, nullptr, nullptr);

         printf("Starting browser is loading %d hasdocument %d\n", browser->IsLoading(), browser->HasDocument());

         printf("Start StartWindow() event loop\n");


         while(browser) {

            CefDoMessageLoopWork();

            // printf("Last browser loading %d hasdocument %d\n", last_browser->IsLoading(), last_browser->HasDocument());

            auto frame = browser->GetMainFrame();

            if (frame && browser->HasDocument() && !browser->IsLoading()) {
               CefRefPtr< CefStringVisitor > visitor = new MyVisitor();
               frame->GetSource(visitor);
               browser = nullptr;

               break; // do once

            }

            gSystem->Sleep(5);
         }

         printf("End StartWindow() loop\n");


      } else {
         CefBrowserHost::CreateBrowser(window_info, fGuiHandler, url, browser_settings, nullptr, nullptr);
      }
#else
      CefBrowserHost::CreateBrowser(window_info, fGuiHandler, url, browser_settings, nullptr);
#endif
   }

}



