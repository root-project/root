/// \file gui_handler.cxx
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

#if !defined(_MSC_VER)
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wshadow"
#endif

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
   PlatformInit();
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
