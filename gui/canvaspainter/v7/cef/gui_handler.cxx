/// \file gui_handler.h
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

#include "gui_handler.h"

#include <sstream>
#include <string>

#include "include/base/cef_bind.h"
#include "include/cef_app.h"
#include "include/views/cef_browser_view.h"
#include "include/views/cef_window.h"
#include "include/wrapper/cef_closure_task.h"
#include "include/wrapper/cef_helpers.h"

GuiHandler::GuiHandler(THttpServer *serv, bool use_views) : BaseHandler(serv), use_views_(use_views)
{
}

GuiHandler::~GuiHandler()
{
}

void GuiHandler::OnTitleChange(CefRefPtr<CefBrowser> browser, const CefString &title)
{
   CEF_REQUIRE_UI_THREAD();

   if (use_views_) {
      // Set the title of the window using the Views framework.
      CefRefPtr<CefBrowserView> browser_view = CefBrowserView::GetForBrowser(browser);
      if (browser_view) {
         CefRefPtr<CefWindow> window = browser_view->GetWindow();
         if (window) window->SetTitle(title);
      }
   } else {
      // Set the title of the window using platform APIs.
      PlatformTitleChange(browser, title);
   }
}

bool GuiHandler::OnConsoleMessage(CefRefPtr<CefBrowser> browser, const CefString &message, const CefString &source,
                                  int line)
{
   printf("CONSOLE: %s\n", message.ToString().c_str());

   // CefRefPtr<CefProcessMessage> msg = CefProcessMessage::Create("PING");
   // Send the process message to the render process.
   // Use PID_BROWSER instead when sending a message to the browser process.
   // browser->SendProcessMessage(PID_RENDERER, msg);

   return true;
}
