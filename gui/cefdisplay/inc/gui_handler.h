/// \file gui_handler.h
/// \ingroup WebGui
/// \author Sergey Linev <S.Linev@gsi.de>
/// \date 2017-06-29
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_cef_gui_handler_h
#define ROOT_cef_gui_handler_h

#include "include/cef_client.h"
#include <list>

class THttpServer;

class GuiHandler : public CefClient, public CefLifeSpanHandler, public CefLoadHandler, public CefDisplayHandler {
protected:
   THttpServer *fServer{nullptr};
   bool fUseViews{false};
   int fConsole{0};
   // List of existing browser windows. Only accessed on the CEF UI thread.
   typedef std::list<CefRefPtr<CefBrowser>> BrowserList;
   BrowserList browser_list_;

   bool is_closing_;

public:
   explicit GuiHandler(THttpServer *serv = nullptr, bool use_views = false);

   // Provide access to the single global instance of this object.
   // static BaseHandler *GetInstance();

   // CefClient methods:
   virtual CefRefPtr<CefLifeSpanHandler> GetLifeSpanHandler() OVERRIDE { return this; }
   virtual CefRefPtr<CefLoadHandler> GetLoadHandler() OVERRIDE { return this; }
   virtual CefRefPtr<CefDisplayHandler> GetDisplayHandler() OVERRIDE { return this; }

   // CefLifeSpanHandler methods:
   virtual void OnAfterCreated(CefRefPtr<CefBrowser> browser) OVERRIDE;
   virtual bool DoClose(CefRefPtr<CefBrowser> browser) OVERRIDE;
   virtual void OnBeforeClose(CefRefPtr<CefBrowser> browser) OVERRIDE;

   // CefLoadHandler methods:
   virtual void OnLoadError(CefRefPtr<CefBrowser> browser, CefRefPtr<CefFrame> frame, ErrorCode errorCode,
                            const CefString &errorText, const CefString &failedUrl) OVERRIDE;

   // CefDisplayHandler methods:
   virtual void OnTitleChange(CefRefPtr<CefBrowser> browser, const CefString &title) OVERRIDE;

   virtual bool OnConsoleMessage(CefRefPtr<CefBrowser> browser,
                                 cef_log_severity_t level,
                                 const CefString &message, const CefString &source,
                                 int line) OVERRIDE;

   // Request that all existing browser windows close.
   void CloseAllBrowsers(bool force_close);

   bool IsClosing() const { return is_closing_; }

private:

   // Platform-specific initialization
   void PlatformInit();

   // Platform-specific implementation.
   void PlatformTitleChange(CefRefPtr<CefBrowser> browser, const CefString &title);

   // Include the default reference counting implementation.
   IMPLEMENT_REFCOUNTING(GuiHandler);
   DISALLOW_COPY_AND_ASSIGN(GuiHandler);
};


#endif // ROOT_cef_gui_handler_h
