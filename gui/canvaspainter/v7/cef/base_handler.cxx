/// \file base_handler.cxx
/// \ingroup CanvasPainter ROOT7
/// \author Sergey Linev <S.Linev@gsi.de>
/// \date 2017-06-29
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

#include "base_handler.h"

#include "THttpServer.h"
#include "THttpEngine.h"
#include "THttpCallArg.h"
#include "TRootSniffer.h"
#include "TString.h"

#include <sstream>
#include <string>

#include "include/base/cef_bind.h"
#include "include/cef_app.h"
#include "include/views/cef_browser_view.h"
#include "include/views/cef_window.h"
#include "include/wrapper/cef_closure_task.h"
#include "include/wrapper/cef_helpers.h"

namespace {

//////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TCefWSEngine                                                         //
//                                                                      //
// Emulation of websocket with CEF messages                             //
// Allows to send data from ROOT server to JS client without            //
// involving special HTTP requests                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TCefWSEngine : public THttpWSEngine {
protected:
   CefRefPtr<CefMessageRouterBrowserSide::Callback> fCallback;

public:
   TCefWSEngine(const char *name, const char *title, CefRefPtr<CefMessageRouterBrowserSide::Callback> callback)
      : THttpWSEngine(name, title), fCallback(callback)
   {
   }

   virtual ~TCefWSEngine() {}

   virtual UInt_t GetId() const
   {
      const void *ptr = (const void *)this;
      return TString::Hash((void *)&ptr, sizeof(void *));
   }

   virtual void ClearHandle() { fCallback->Failure(0, "close"); }

   virtual void Send(const void * /*buf*/, int /*len*/)
   {
      Error("TLongPollEngine::Send", "Should never be called, only text is supported");
   }

   virtual void SendCharStar(const char *buf)
   {
      // printf("CEF sends message to client %s\n", buf);
      fCallback->Success(buf); // send next message to JS
   }

   virtual Bool_t PreviewData(THttpCallArg *arg)
   {
      // function called in the user code before processing correspondent websocket data
      // returns kTRUE when user should ignore such http request - it is for internal use

      // this is normal request, deliver and process it as any other
      return kFALSE;
   }
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TCefWsCallArg                                                        //
//                                                                      //
// HTTp call argument provided to http server from CEF messaging        //
// Allows immediately reply to JS client                                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TCefWsCallArg : public THttpCallArg {
protected:
   CefRefPtr<CefMessageRouterBrowserSide::Callback> fCallback;

public:
   TCefWsCallArg(CefRefPtr<CefMessageRouterBrowserSide::Callback> callback) : fCallback(callback) {}

   virtual void HttpReplied()
   {
      if (fCallback == NULL) return;

      if (Is404()) {
         fCallback->Failure(0, "error");
      } else {
         std::string reply;
         if (GetContentLength() > 0) reply.append((const char *)GetContent(), GetContentLength());
         fCallback->Success(reply);
      }
      fCallback = NULL;
   }
};

// Handle messages in the browser process.
class RootMessageHandler : public CefMessageRouterBrowserSide::Handler {
protected:
   THttpServer *fServer;

public:
   explicit RootMessageHandler(THttpServer *serv = 0) : fServer(serv) {}

   // Called due to cefQuery execution in message_router.html.
   bool OnQuery(CefRefPtr<CefBrowser> browser, CefRefPtr<CefFrame> frame, int64 query_id, const CefString &request,
                bool persistent, CefRefPtr<Callback> callback) OVERRIDE
   {
      std::string message = request;

      if (message == "init_jsroot_done") {
         printf("Get message %s\n", message.c_str());
         std::string result = "confirm from ROOT";
         callback->Success(result);
         return true; // processed
      }

      if (!fServer) return false;

      // message format
      // <path>::connect, replied with ws handler id
      // <path>::<connid>::post::<data> or <path>::<connid>::close

      int pos = message.find("::");
      if (pos == std::string::npos) return false;

      std::string url = message.substr(0, pos);
      message.erase(0, pos + 2);

      TCefWsCallArg *arg = new TCefWsCallArg(callback);
      arg->SetPathName(url.c_str());
      arg->SetFileName("root.ws_emulation");

      if (message == "connect") {
         TCefWSEngine *ws = new TCefWSEngine("name", "title", callback);
         arg->SetMethod("WS_CONNECT");
         arg->SetWSHandle(ws);
         arg->SetWSId(ws->GetId());
         printf("Create CEF WS engine with id %u\n", ws->GetId());
      } else {
         pos = message.find("::");
         if (pos == std::string::npos) return false;
         std::string sid = message.substr(0, pos);
         message.erase(0, pos + 2);
         unsigned wsid = 0;
         sscanf(sid.c_str(), "%u", &wsid);
         arg->SetWSId(wsid);
         if (message == "close") {
            arg->SetMethod("WS_CLOSE");
         } else {
            arg->SetMethod("WS_DATA");
            if (message.length() > 6) arg->SetPostData((void *)(message.c_str() + 6), message.length() - 6, kTRUE);
         }
      }

      if (fServer->SubmitHttp(arg, kTRUE)) arg->HttpReplied(); // message processed and can be replied

      return true;
   }

private:
   DISALLOW_COPY_AND_ASSIGN(RootMessageHandler);
};

BaseHandler *g_instance = NULL;

} // namespace

BaseHandler::BaseHandler(THttpServer *serv) : fServer(serv), is_closing_(false)
{
   DCHECK(!g_instance);
   g_instance = this;
}

BaseHandler::~BaseHandler()
{
   g_instance = NULL;
}

// static
BaseHandler *BaseHandler::GetInstance()
{
   return g_instance;
}

bool BaseHandler::OnProcessMessageReceived(CefRefPtr<CefBrowser> browser, CefProcessId source_process,
                                           CefRefPtr<CefProcessMessage> message)
{
   CEF_REQUIRE_UI_THREAD();

   return message_router_->OnProcessMessageReceived(browser, source_process, message);
}

void BaseHandler::OnAfterCreated(CefRefPtr<CefBrowser> browser)
{
   CEF_REQUIRE_UI_THREAD();

   if (!message_router_) {
      // Create the browser-side router for query handling.
      CefMessageRouterConfig config;
      message_router_ = CefMessageRouterBrowserSide::Create(config);

      // Register handlers with the router.
      message_handler_.reset(new RootMessageHandler(fServer));
      message_router_->AddHandler(message_handler_.get(), false);
   }

   // Add to the list of existing browsers.
   browser_list_.push_back(browser);
}

bool BaseHandler::OnBeforeBrowse(CefRefPtr<CefBrowser> browser, CefRefPtr<CefFrame> frame,
                                 CefRefPtr<CefRequest> request, bool is_redirect)
{
   CEF_REQUIRE_UI_THREAD();

   message_router_->OnBeforeBrowse(browser, frame);
   return false;
}

void BaseHandler::OnRenderProcessTerminated(CefRefPtr<CefBrowser> browser, TerminationStatus status)
{
   CEF_REQUIRE_UI_THREAD();

   message_router_->OnRenderProcessTerminated(browser);
}

bool BaseHandler::DoClose(CefRefPtr<CefBrowser> browser)
{
   CEF_REQUIRE_UI_THREAD();

   // Closing the main window requires special handling. See the DoClose()
   // documentation in the CEF header for a detailed destription of this
   // process.
   if (browser_list_.size() == 1) {
      // Set a flag to indicate that the window close should be allowed.
      is_closing_ = true;
   }

   // Allow the close. For windowed browsers this will result in the OS close
   // event being sent.
   return false;
}

void BaseHandler::OnBeforeClose(CefRefPtr<CefBrowser> browser)
{
   CEF_REQUIRE_UI_THREAD();

   // Remove from the list of existing browsers.
   BrowserList::iterator bit = browser_list_.begin();
   for (; bit != browser_list_.end(); ++bit) {
      if ((*bit)->IsSame(browser)) {
         browser_list_.erase(bit);
         break;
      }
   }

   if (browser_list_.empty()) {

      message_router_->RemoveHandler(message_handler_.get());
      message_handler_.reset();
      message_router_ = NULL;

      // All browser windows have closed. Quit the application message loop.

      CefQuitMessageLoop();
   }
}

void BaseHandler::OnLoadError(CefRefPtr<CefBrowser> browser, CefRefPtr<CefFrame> frame, ErrorCode errorCode,
                              const CefString &errorText, const CefString &failedUrl)
{
   CEF_REQUIRE_UI_THREAD();

   // Don't display an error for downloaded files.
   if (errorCode == ERR_ABORTED) return;

   // Display a load error message.
   std::stringstream ss;
   ss << "<html><body bgcolor=\"white\">"
         "<h2>Failed to load URL "
      << std::string(failedUrl) << " with error " << std::string(errorText) << " (" << errorCode
      << ").</h2></body></html>";
   frame->LoadString(ss.str(), failedUrl);
}

void BaseHandler::CloseAllBrowsers(bool force_close)
{
   if (!CefCurrentlyOn(TID_UI)) {
      // Execute on the UI thread.
      CefPostTask(TID_UI, base::Bind(&BaseHandler::CloseAllBrowsers, this, force_close));
      return;
   }

   if (browser_list_.empty()) return;

   BrowserList::const_iterator it = browser_list_.begin();
   for (; it != browser_list_.end(); ++it) (*it)->GetHost()->CloseBrowser(force_close);
}
