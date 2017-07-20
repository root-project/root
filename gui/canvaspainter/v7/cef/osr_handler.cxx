// Copyright (c) 2013 The Chromium Embedded Framework Authors. All rights
// reserved. Use of this source code is governed by a BSD-style license that
// can be found in the LICENSE file.

#include "osr_handler.h"

#include <sstream>
#include <string>

#include "include/base/cef_bind.h"
#include "include/cef_app.h"
#include "include/views/cef_browser_view.h"
#include "include/views/cef_window.h"
#include "include/wrapper/cef_closure_task.h"
#include "include/wrapper/cef_helpers.h"

namespace {

OsrHandler *g_instance = NULL;

} // namespace

OsrHandler::OsrHandler() : is_closing_(false)
{
   DCHECK(!g_instance);
   g_instance = this;
}

OsrHandler::~OsrHandler()
{
   g_instance = NULL;
}

// static
OsrHandler *OsrHandler::GetInstance()
{
   return g_instance;
}

void OsrHandler::OnTitleChange(CefRefPtr<CefBrowser> browser, const CefString &title)
{
   CEF_REQUIRE_UI_THREAD();
}

bool OsrHandler::OnConsoleMessage(CefRefPtr<CefBrowser> browser, const CefString &message, const CefString &source,
                                  int line)
{
   printf("CONSOLE: %s\n", message.ToString().c_str());
   return true;
}

void OsrHandler::OnAfterCreated(CefRefPtr<CefBrowser> browser)
{
   CEF_REQUIRE_UI_THREAD();

   // Add to the list of existing browsers.
   browser_list_.push_back(browser);
}

bool OsrHandler::DoClose(CefRefPtr<CefBrowser> browser)
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

void OsrHandler::OnBeforeClose(CefRefPtr<CefBrowser> browser)
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
      // All browser windows have closed. Quit the application message loop.
      CefQuitMessageLoop();
   }
}

void OsrHandler::OnLoadError(CefRefPtr<CefBrowser> browser, CefRefPtr<CefFrame> frame, ErrorCode errorCode,
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

void OsrHandler::CloseAllBrowsers(bool force_close)
{
   if (!CefCurrentlyOn(TID_UI)) {
      // Execute on the UI thread.
      CefPostTask(TID_UI, base::Bind(&OsrHandler::CloseAllBrowsers, this, force_close));
      return;
   }

   if (browser_list_.empty()) return;

   BrowserList::const_iterator it = browser_list_.begin();
   for (; it != browser_list_.end(); ++it) (*it)->GetHost()->CloseBrowser(force_close);
}

bool OsrHandler::GetRootScreenRect(CefRefPtr<CefBrowser> browser, CefRect &rect)
{
   CEF_REQUIRE_UI_THREAD();

   return false;

   // if (!osr_delegate_) return false;
   // return osr_delegate_->GetRootScreenRect(browser, rect);
}

bool OsrHandler::GetViewRect(CefRefPtr<CefBrowser> browser, CefRect &rect)
{
   CEF_REQUIRE_UI_THREAD();

   return false;

   // if (!osr_delegate_) return false;
   // return osr_delegate_->GetViewRect(browser, rect);
}

bool OsrHandler::GetScreenPoint(CefRefPtr<CefBrowser> browser, int viewX, int viewY, int &screenX, int &screenY)
{
   CEF_REQUIRE_UI_THREAD();

   return false;
   // if (!osr_delegate_) return false;
   // return osr_delegate_->GetScreenPoint(browser, viewX, viewY, screenX, screenY);
}

bool OsrHandler::GetScreenInfo(CefRefPtr<CefBrowser> browser, CefScreenInfo &screen_info)
{
   CEF_REQUIRE_UI_THREAD();

   return false;

   // if (!osr_delegate_) return false;
   // return osr_delegate_->GetScreenInfo(browser, screen_info);
}

void OsrHandler::OnPopupShow(CefRefPtr<CefBrowser> browser, bool show)
{
   CEF_REQUIRE_UI_THREAD();

   // if (!osr_delegate_) return;
   // return osr_delegate_->OnPopupShow(browser, show);
}

void OsrHandler::OnPopupSize(CefRefPtr<CefBrowser> browser, const CefRect &rect)
{
   CEF_REQUIRE_UI_THREAD();
   // if (!osr_delegate_) return;
   // return osr_delegate_->OnPopupSize(browser, rect);
}

void OsrHandler::OnPaint(CefRefPtr<CefBrowser> browser, PaintElementType type, const RectList &dirtyRects,
                         const void *buffer, int width, int height)
{
   CEF_REQUIRE_UI_THREAD();
   // if (!osr_delegate_) return;
   // osr_delegate_->OnPaint(browser, type, dirtyRects, buffer, width, height);
}

void OsrHandler::OnCursorChange(CefRefPtr<CefBrowser> browser, CefCursorHandle cursor, CursorType type,
                                const CefCursorInfo &custom_cursor_info)
{
   CEF_REQUIRE_UI_THREAD();
   // if (!osr_delegate_) return;
   // osr_delegate_->OnCursorChange(browser, cursor, type, custom_cursor_info);
}

bool OsrHandler::StartDragging(CefRefPtr<CefBrowser> browser, CefRefPtr<CefDragData> drag_data,
                               CefRenderHandler::DragOperationsMask allowed_ops, int x, int y)
{
   CEF_REQUIRE_UI_THREAD();

   return false;

   // if (!osr_delegate_) return false;
   // return osr_delegate_->StartDragging(browser, drag_data, allowed_ops, x, y);
}

void OsrHandler::UpdateDragCursor(CefRefPtr<CefBrowser> browser, CefRenderHandler::DragOperation operation)
{
   CEF_REQUIRE_UI_THREAD();

   // if (!osr_delegate_) return;
   // osr_delegate_->UpdateDragCursor(browser, operation);
}

void OsrHandler::OnImeCompositionRangeChanged(CefRefPtr<CefBrowser> browser, const CefRange &selection_range,
                                              const CefRenderHandler::RectList &character_bounds)
{
   CEF_REQUIRE_UI_THREAD();
   // if (!osr_delegate_) return;
   // osr_delegate_->OnImeCompositionRangeChanged(browser, selection_range, character_bounds);
}
