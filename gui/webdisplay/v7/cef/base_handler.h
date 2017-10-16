/// \file base_handler.h
/// \ingroup CanvasPainter ROOT7
/// \author Sergey Linev <S.Linev@gsi.de>
/// \date 2017-06-29
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_cef_base_handler
#define ROOT_cef_base_handler

#include "include/cef_client.h"

#include "include/wrapper/cef_message_router.h"

#include <list>

class THttpServer;

/// Class used to handle off-screen application and should emulate some render requests

class BaseHandler : public CefClient, public CefLifeSpanHandler, public CefLoadHandler, public CefRequestHandler {
protected:
   THttpServer *fServer;

   // Handles the browser side of query routing.
   CefRefPtr<CefMessageRouterBrowserSide> message_router_;
   scoped_ptr<CefMessageRouterBrowserSide::Handler> message_handler_;

public:
   explicit BaseHandler(THttpServer *serv = 0);
   virtual ~BaseHandler();

   // Provide access to the single global instance of this object.
   static BaseHandler *GetInstance();

   // CefClient methods:
   virtual CefRefPtr<CefLifeSpanHandler> GetLifeSpanHandler() OVERRIDE { return this; }
   virtual CefRefPtr<CefLoadHandler> GetLoadHandler() OVERRIDE { return this; }
   virtual CefRefPtr<CefRequestHandler> GetRequestHandler() OVERRIDE { return this; }

   virtual bool OnProcessMessageReceived(CefRefPtr<CefBrowser> browser, CefProcessId source_process,
                                         CefRefPtr<CefProcessMessage> message) OVERRIDE;

   // CefLifeSpanHandler methods:
   virtual void OnAfterCreated(CefRefPtr<CefBrowser> browser) OVERRIDE;
   virtual bool DoClose(CefRefPtr<CefBrowser> browser) OVERRIDE;
   virtual void OnBeforeClose(CefRefPtr<CefBrowser> browser) OVERRIDE;

   // CefLoadHandler methods:
   virtual void OnLoadError(CefRefPtr<CefBrowser> browser, CefRefPtr<CefFrame> frame, ErrorCode errorCode,
                            const CefString &errorText, const CefString &failedUrl) OVERRIDE;

   // CefRequestHandler methods:
   virtual bool OnBeforeBrowse(CefRefPtr<CefBrowser> browser, CefRefPtr<CefFrame> frame, CefRefPtr<CefRequest> request,
                               bool is_redirect) OVERRIDE;
   virtual void OnRenderProcessTerminated(CefRefPtr<CefBrowser> browser, TerminationStatus status) OVERRIDE;

   // Request that all existing browser windows close.
   void CloseAllBrowsers(bool force_close);

   bool IsClosing() const { return is_closing_; }

private:
   // List of existing browser windows. Only accessed on the CEF UI thread.
   typedef std::list<CefRefPtr<CefBrowser>> BrowserList;
   BrowserList browser_list_;

   bool is_closing_;

   // Include the default reference counting implementation.
   IMPLEMENT_REFCOUNTING(BaseHandler);
   DISALLOW_COPY_AND_ASSIGN(BaseHandler);
};

#endif // ROOT_cef_osr_handler
