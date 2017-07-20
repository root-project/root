// Copyright (c) 2013 The Chromium Embedded Framework Authors. All rights
// reserved. Use of this source code is governed by a BSD-style license that
// can be found in the LICENSE file.

#ifndef ROOT_cef_osr_handler
#define ROOT_cef_osr_handler

#include "include/cef_client.h"

#include "include/wrapper/cef_message_router.h"

#include <list>

class OsrHandler : public CefClient,
                   public CefDisplayHandler,
                   public CefLifeSpanHandler,
                   public CefLoadHandler,
                   public CefRenderHandler,
                   public CefRequestHandler {
protected:
   // Handles the browser side of query routing.
   CefRefPtr<CefMessageRouterBrowserSide> message_router_;
   scoped_ptr<CefMessageRouterBrowserSide::Handler> message_handler_;

public:
   explicit OsrHandler();
   ~OsrHandler();

   // Provide access to the single global instance of this object.
   static OsrHandler *GetInstance();

   // CefClient methods:
   virtual CefRefPtr<CefDisplayHandler> GetDisplayHandler() OVERRIDE { return this; }
   virtual CefRefPtr<CefLifeSpanHandler> GetLifeSpanHandler() OVERRIDE { return this; }
   virtual CefRefPtr<CefLoadHandler> GetLoadHandler() OVERRIDE { return this; }
   virtual CefRefPtr<CefRenderHandler> GetRenderHandler() OVERRIDE { return this; }
   virtual CefRefPtr<CefRequestHandler> GetRequestHandler() OVERRIDE { return this; }

   virtual bool OnProcessMessageReceived(CefRefPtr<CefBrowser> browser, CefProcessId source_process,
                                         CefRefPtr<CefProcessMessage> message) OVERRIDE;

   // CefDisplayHandler methods:
   virtual void OnTitleChange(CefRefPtr<CefBrowser> browser, const CefString &title) OVERRIDE;

   // CefLifeSpanHandler methods:
   virtual void OnAfterCreated(CefRefPtr<CefBrowser> browser) OVERRIDE;
   virtual bool DoClose(CefRefPtr<CefBrowser> browser) OVERRIDE;
   virtual void OnBeforeClose(CefRefPtr<CefBrowser> browser) OVERRIDE;

   // CefLoadHandler methods:
   virtual void OnLoadError(CefRefPtr<CefBrowser> browser, CefRefPtr<CefFrame> frame, ErrorCode errorCode,
                            const CefString &errorText, const CefString &failedUrl) OVERRIDE;

   virtual bool OnConsoleMessage(CefRefPtr<CefBrowser> browser, const CefString &message, const CefString &source,
                                 int line) OVERRIDE;

   // CefRequestHandler methods:
   virtual bool OnBeforeBrowse(CefRefPtr<CefBrowser> browser, CefRefPtr<CefFrame> frame, CefRefPtr<CefRequest> request,
                               bool is_redirect) OVERRIDE;
   virtual void OnRenderProcessTerminated(CefRefPtr<CefBrowser> browser, TerminationStatus status) OVERRIDE;

   // CefRenderHandler methods.
   virtual bool GetRootScreenRect(CefRefPtr<CefBrowser> browser, CefRect &rect) OVERRIDE;
   virtual bool GetViewRect(CefRefPtr<CefBrowser> browser, CefRect &rect) OVERRIDE;
   virtual bool GetScreenPoint(CefRefPtr<CefBrowser> browser, int viewX, int viewY, int &screenX,
                               int &screenY) OVERRIDE;
   virtual bool GetScreenInfo(CefRefPtr<CefBrowser> browser, CefScreenInfo &screen_info) OVERRIDE;
   virtual void OnPopupShow(CefRefPtr<CefBrowser> browser, bool show) OVERRIDE;
   virtual void OnPopupSize(CefRefPtr<CefBrowser> browser, const CefRect &rect) OVERRIDE;
   virtual void OnPaint(CefRefPtr<CefBrowser> browser, CefRenderHandler::PaintElementType type,
                        const CefRenderHandler::RectList &dirtyRects, const void *buffer, int width,
                        int height) OVERRIDE;
   virtual void OnCursorChange(CefRefPtr<CefBrowser> browser, CefCursorHandle cursor, CursorType type,
                               const CefCursorInfo &custom_cursor_info) OVERRIDE;
   virtual bool StartDragging(CefRefPtr<CefBrowser> browser, CefRefPtr<CefDragData> drag_data,
                              CefRenderHandler::DragOperationsMask allowed_ops, int x, int y) OVERRIDE;
   virtual void UpdateDragCursor(CefRefPtr<CefBrowser> browser, CefRenderHandler::DragOperation operation) OVERRIDE;
   virtual void OnImeCompositionRangeChanged(CefRefPtr<CefBrowser> browser, const CefRange &selection_range,
                                             const CefRenderHandler::RectList &character_bounds) OVERRIDE;

   // Request that all existing browser windows close.
   void CloseAllBrowsers(bool force_close);

   bool IsClosing() const { return is_closing_; }

private:
   // List of existing browser windows. Only accessed on the CEF UI thread.
   typedef std::list<CefRefPtr<CefBrowser>> BrowserList;
   BrowserList browser_list_;

   bool is_closing_;

   // Include the default reference counting implementation.
   IMPLEMENT_REFCOUNTING(OsrHandler);
   DISALLOW_COPY_AND_ASSIGN(OsrHandler);
};

#endif // CEF_TESTS_CEFSIMPLE_SIMPLE_HANDLER_H_
