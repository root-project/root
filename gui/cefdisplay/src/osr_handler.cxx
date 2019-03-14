/// \file osr_handler.cxx
/// \ingroup CanvasPainter ROOT7
/// \author Sergey Linev <S.Linev@gsi.de>
/// \date 2017-06-29
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

#if !defined(_MSC_VER)
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wshadow"
#endif

#include "osr_handler.h"

#include <sstream>
#include <string>

#include "include/base/cef_bind.h"
#include "include/cef_app.h"
#include "include/views/cef_browser_view.h"
#include "include/views/cef_window.h"
#include "include/wrapper/cef_closure_task.h"
#include "include/wrapper/cef_helpers.h"

OsrHandler::OsrHandler(THttpServer *serv) : BaseHandler(serv)
{
}

bool OsrHandler::GetRootScreenRect(CefRefPtr<CefBrowser> browser, CefRect &rect)
{
   CEF_REQUIRE_UI_THREAD();

   rect.x = rect.y = 0;
   rect.width = 800;
   rect.height = 600;

   return true;

   // if (!osr_delegate_) return false;
   // return osr_delegate_->GetRootScreenRect(browser, rect);
}

#if CEF_COMMIT_NUMBER > 1894
void OsrHandler::GetViewRect(CefRefPtr<CefBrowser> browser, CefRect &rect)
#else
bool OsrHandler::GetViewRect(CefRefPtr<CefBrowser> browser, CefRect &rect)
#endif
{
   CEF_REQUIRE_UI_THREAD();

   rect.x = rect.y = 0;
   rect.width = 800;
   rect.height = 600;

#if CEF_COMMIT_NUMBER <= 1894
   return true;
#endif
   // if (!osr_delegate_) return false;
   // return osr_delegate_->GetViewRect(browser, rect);
}

bool OsrHandler::GetScreenPoint(CefRefPtr<CefBrowser> browser, int viewX, int viewY, int &screenX, int &screenY)
{
   CEF_REQUIRE_UI_THREAD();

   screenX = viewX;
   screenY = viewY;

   return true;
   // if (!osr_delegate_) return false;
   // return osr_delegate_->GetScreenPoint(browser, viewX, viewY, screenX, screenY);
}

bool OsrHandler::GetScreenInfo(CefRefPtr<CefBrowser> browser, CefScreenInfo &screen_info)
{
   CEF_REQUIRE_UI_THREAD();

   CefRect view_rect;
   GetViewRect(browser, view_rect);

   screen_info.device_scale_factor = 1.;

   // The screen info rectangles are used by the renderer to create and position
   // popups. Keep popups inside the view rectangle.
   screen_info.rect = view_rect;
   screen_info.available_rect = view_rect;
   return true;

   // if (!osr_delegate_) return false;
   // return osr_delegate_->GetScreenInfo(browser, screen_info);
}

void OsrHandler::OnPopupShow(CefRefPtr<CefBrowser> browser, bool show)
{
   CEF_REQUIRE_UI_THREAD();

   // do nothing

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

   // REQUIRE_MAIN_THREAD(); // we can request to run code in main thread?

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
